"""
Flask web application for managing whitelist dataset, training model, testing URLs, and collecting RLHF feedback.

This application provides a REST API for:
- Managing whitelist entries used in training the ad-blocker AI model
- Triggering model training and viewing results
- Testing URLs with the trained model
- Collecting RLHF (Reinforcement Learning from Human Feedback) data

It integrates with AdBlockListParser, AdBlockerModel, and FeedbackCollector classes.
"""

import os
import sys
import subprocess
import threading
import json
import urllib.parse
import re
from flask import Flask, request, jsonify, render_template
from typing import Optional, Set, Dict, Any
from datetime import datetime
from adblocker_ai import AdBlockListParser, AdBlockerModel, FeedbackCollector, F1Score

app = Flask(__name__)

# Configuration
WHITELIST_FILE = 'whitelist_dataset.txt'
FEEDBACK_FILE = 'feedback_data.json'
MODEL_PATH = './adblocker_model.keras'
DEFAULT_MODEL_PATH = './adblocker_model'

# Initialize parser
parser = AdBlockListParser()

# Global model instance (lazy-loaded)
model: Optional[AdBlockerModel] = None
model_lock = threading.RLock()  # Use RLock (reentrant lock) to prevent deadlocks

# Global feedback collector
feedback_collector: Optional[FeedbackCollector] = None

# Training status
training_status = {
    'status': 'idle',  # 'idle', 'running', 'completed', 'error'
    'results': None,
    'output': None,
    'error': None
}
training_lock = threading.Lock()


def extract_domain(url_or_domain: str) -> Optional[str]:
    """
    Extract domain from URL or return domain as-is.
    
    Args:
        url_or_domain: URL string (e.g., "https://example.com/path") or domain (e.g., "example.com")
        
    Returns:
        Extracted domain string, or None if invalid
    """
    if not url_or_domain or not url_or_domain.strip():
        return None
    
    url_or_domain = url_or_domain.strip()
    
    # Try parsing as URL first
    try:
        parsed = urllib.parse.urlparse(url_or_domain)
        if parsed.netloc:
            domain = parsed.netloc
        elif parsed.path and not url_or_domain.startswith('http'):
            # Might be a domain without protocol
            domain = parsed.path.split('/')[0]
        else:
            domain = url_or_domain.split('/')[0].split(':')[0]
    except Exception:
        # If parsing fails, treat as domain
        domain = url_or_domain.split('/')[0].split(':')[0]
    
    # Remove port if present
    domain = domain.split(':')[0]
    
    # Basic domain validation (contains at least one dot, or is a valid TLD)
    if not domain or ('.' not in domain and domain not in ['com', 'org', 'net', 'edu', 'gov', 'io', 'co', 'me', 'tv', 'info']):
        return None
    
    # Normalize: remove www. prefix
    if domain.lower().startswith('www.'):
        domain = domain[4:]
    
    return domain.lower()


def validate_domain(domain: str) -> bool:
    """
    Validate domain format.
    
    Args:
        domain: Domain string to validate
        
    Returns:
        True if domain is valid, False otherwise
    """
    if not domain or not domain.strip():
        return False
    
    domain = domain.strip().lower()
    
    # Check for valid characters (alphanumeric, dots, hyphens)
    if not re.match(r'^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)*$', domain):
        return False
    
    # Must contain at least one dot (except for single TLDs)
    if '.' not in domain and domain not in ['com', 'org', 'net', 'edu', 'gov', 'io', 'co', 'me', 'tv', 'info']:
        return False
    
    return True


def load_whitelist_from_file() -> Set[str]:
    """
    Load whitelist domains from file.
    
    Returns:
        Set of domain strings
    """
    if not os.path.exists(WHITELIST_FILE):
        return set()
    
    try:
        with open(WHITELIST_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        domains = parser.parse_plain_domain_list(content)
        
        # Update parser's whitelist
        parser.whitelist_domains.clear()
        for domain in domains:
            parser.add_whitelist_domain(domain)
        
        return domains
    except Exception as e:
        print(f"Error loading whitelist from file: {e}")
        return set()


def save_whitelist_to_file():
    """
    Save whitelist domains to file (one domain per line).
    """
    try:
        domains = sorted(parser.get_whitelist_domains())
        with open(WHITELIST_FILE, 'w', encoding='utf-8') as f:
            for domain in domains:
                f.write(f"{domain}\n")
    except Exception as e:
        print(f"Error saving whitelist to file: {e}")
        raise


# Load whitelist on module import
_whitelist_loaded = False

def ensure_whitelist_loaded():
    """Ensure whitelist is loaded from file."""
    global _whitelist_loaded
    if not _whitelist_loaded:
        load_whitelist_from_file()
        _whitelist_loaded = True

# Load whitelist when module is imported
ensure_whitelist_loaded()

# Initialize feedback collector
feedback_collector = FeedbackCollector(storage_path=FEEDBACK_FILE)


def load_model_instance() -> Optional[AdBlockerModel]:
    """
    Load the trained model instance (lazy loading, cached).
    
    Returns:
        AdBlockerModel instance or None if model not found
    """
    global model
    
    acquired = model_lock.acquire(blocking=False)
    if not acquired:
        model_lock.acquire(blocking=True)
    
    try:
        # Check if cached model is valid
        if model is not None:
            # Verify the cached model is actually loaded
            if hasattr(model, 'model') and model.model is not None and getattr(model, 'is_trained', False):
                return model
            else:
                # Clear the invalid cached model
                model = None
        
        # Try loading model from paths - prefer .keras file over directory
        model_paths = [MODEL_PATH, DEFAULT_MODEL_PATH]
        
        for path in model_paths:
            # Only try files, not directories (Keras 3 can't load from directories)
            if os.path.exists(path) and os.path.isfile(path):
                try:
                    model = AdBlockerModel(input_dim=25, whitelist_parser=parser)
                    model.load_model(path)
                    # Verify the model was actually loaded
                    if model.model is None:
                        raise ValueError(f"Model failed to load from {path} - model.model is None")
                    return model
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")
                    continue
        
        return None
    finally:
        model_lock.release()


def reload_model():
    """Force reload the model (useful after training)."""
    global model
    with model_lock:
        model = None
    # Call load_model_instance outside the lock to avoid potential deadlock
    load_model_instance()


def run_training_thread(epochs: int = 50, batch_size: int = 32, max_samples: Optional[int] = None):
    """
    Run training in a background thread.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        max_samples: Maximum samples per class
    """
    global training_status
    
    with training_lock:
        training_status['status'] = 'running'
        training_status['results'] = None
        training_status['output'] = ''
        training_status['error'] = None
    
    try:
        # Build command
        cmd = [sys.executable, 'adblocker_ai.py',
               '--epochs', str(epochs),
               '--batch-size', str(batch_size)]
        
        if max_samples:
            cmd.extend(['--max-samples', str(max_samples)])
        
        # Run subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            with training_lock:
                training_status['output'] = ''.join(output_lines)
        
        process.wait()
        
        # Parse output for results (basic parsing)
        output_text = ''.join(output_lines)
        results = parse_training_output(output_text)
        
        with training_lock:
            training_status['status'] = 'completed' if process.returncode == 0 else 'error'
            training_status['results'] = results
            training_status['output'] = output_text
            if process.returncode != 0:
                training_status['error'] = f'Training failed with exit code {process.returncode}'
        
        # Reload model after training
        if process.returncode == 0:
            reload_model()
    
    except Exception as e:
        with training_lock:
            training_status['status'] = 'error'
            training_status['error'] = str(e)


def parse_training_output(output: str) -> Dict[str, Any]:
    """
    Parse training output to extract metrics and results.
    
    Args:
        output: Training output text
        
    Returns:
        Dictionary with parsed results
    """
    results = {
        'metrics': {},
        'predictions': [],
        'summary': {}
    }
    
    # Parse test set metrics
    metrics_pattern = r'(\w+):\s*([\d.]+)'
    in_metrics_section = False
    
    for line in output.split('\n'):
        if 'Test set metrics:' in line or 'Test set evaluation' in line:
            in_metrics_section = True
            continue
        
        if in_metrics_section:
            matches = re.findall(metrics_pattern, line)
            for metric, value in matches:
                try:
                    results['metrics'][metric] = float(value)
                except ValueError:
                    pass
            
            if not line.strip() or '=' in line:
                in_metrics_section = False
        
        # Parse sample predictions
        if 'Prediction:' in line and 'confidence:' in line:
            parts = line.split('Prediction:')
            if len(parts) == 2:
                url_match = re.search(r'https?://[^\s]+', line)
                pred_match = re.search(r'Prediction:\s*(\w+)', line)
                conf_match = re.search(r'confidence:\s*([\d.]+)', line)
                
                if url_match and pred_match and conf_match:
                    results['predictions'].append({
                        'url': url_match.group(0),
                        'prediction': pred_match.group(1),
                        'confidence': float(conf_match.group(1))
                    })
        
        # Parse summary statistics
        if 'Total parsed data:' in line or 'Generated dataset' in line:
            count_pattern = r'(\w+):\s*(\d+)'
            matches = re.findall(count_pattern, line)
            for key, value in matches:
                try:
                    results['summary'][key] = int(value)
                except ValueError:
                    pass
    
    return results


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/whitelist', methods=['GET'])
def get_whitelist():
    """
    Retrieve all whitelist entries.
    
    Returns:
        JSON response with list of domains
    """
    domains = sorted(list(parser.get_whitelist_domains()))
    return jsonify({
        'success': True,
        'domains': domains,
        'count': len(domains)
    })


@app.route('/api/whitelist', methods=['POST'])
def add_whitelist_entry():
    """
    Add a new URL/domain to the whitelist.
    
    Request body should contain:
        - url_or_domain: URL or domain string to add
    
    Returns:
        JSON response with success status and domain
    """
    data = request.get_json()
    
    if not data or 'url_or_domain' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing required field: url_or_domain'
        }), 400
    
    url_or_domain = data['url_or_domain']
    
    # Extract domain from URL
    domain = extract_domain(url_or_domain)
    
    if not domain:
        return jsonify({
            'success': False,
            'error': f'Invalid URL or domain: {url_or_domain}'
        }), 400
    
    # Validate domain format
    if not validate_domain(domain):
        return jsonify({
            'success': False,
            'error': f'Invalid domain format: {domain}'
        }), 400
    
    # Check if domain already exists
    if domain in parser.whitelist_domains:
        return jsonify({
            'success': False,
            'error': f'Domain already exists in whitelist: {domain}'
        }), 400
    
    # Add domain to parser
    parser.add_whitelist_domain(domain)
    
    # Save to file
    try:
        save_whitelist_to_file()
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to save whitelist: {str(e)}'
        }), 500
    
    return jsonify({
        'success': True,
        'domain': domain,
        'message': f'Domain {domain} added to whitelist'
    }), 201


@app.route('/api/whitelist/<path:domain>', methods=['DELETE'])
def delete_whitelist_entry(domain):
    """
    Remove a domain from the whitelist.
    
    Args:
        domain: Domain string to remove
    
    Returns:
        JSON response with success status
    """
    # Normalize domain (remove URL encoding if present)
    domain = urllib.parse.unquote(domain)
    domain = domain.lower().strip()
    
    # Validate domain format
    if not validate_domain(domain):
        return jsonify({
            'success': False,
            'error': f'Invalid domain format: {domain}'
        }), 400
    
    # Check if domain exists
    if domain not in parser.whitelist_domains:
        return jsonify({
            'success': False,
            'error': f'Domain not found in whitelist: {domain}'
        }), 404
    
    # Remove domain from parser
    parser.remove_whitelist_domain(domain)
    
    # Save to file
    try:
        save_whitelist_to_file()
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to save whitelist: {str(e)}'
        }), 500
    
    return jsonify({
        'success': True,
        'domain': domain,
        'message': f'Domain {domain} removed from whitelist'
    })


@app.route('/api/whitelist/count', methods=['GET'])
def get_whitelist_count():
    """
    Get count of whitelist entries.
    
    Returns:
        JSON response with count
    """
    count = len(parser.get_whitelist_domains())
    return jsonify({
        'success': True,
        'count': count
    })


@app.route('/api/predict', methods=['POST'])
def predict_url():
    """
    Test a URL with the current model.
    
    Request body should contain:
        - url: URL or domain string to test
    
    Returns:
        JSON response with prediction and confidence
    """
    try:
        # Get JSON data from request
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error parsing JSON: {str(e)}'
            }), 400
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Invalid or missing JSON payload'
            }), 400
        
        if 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: url'
            }), 400
        
        url = data['url'].strip()
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'URL cannot be empty'
            }), 400
        
        # Load model
        model_instance = load_model_instance()
        
        if model_instance is None:
            return jsonify({
                'success': False,
                'error': 'Model not found. Please train the model first.'
            }), 404
        
        # Get prediction
        prediction, confidence = model_instance.predict(url)
        
        label = 'AD' if prediction == 1 else 'LEGITIMATE'
        
        return jsonify({
            'success': True,
            'url': url,
            'prediction': int(prediction),
            'label': label,
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit RLHF feedback for a prediction.
    
    Request body should contain:
        - url: URL or domain that was tested
        - model_prediction: Model's prediction (0 or 1)
        - human_label: Human's label/correction (0 or 1)
        - confidence: Model's confidence score (optional)
        - metadata: Optional additional metadata (dict)
    
    Returns:
        JSON response with success status
    """
    data = request.get_json()
    
    required_fields = ['url', 'model_prediction', 'human_label']
    if not data or not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'error': f'Missing required fields: {", ".join(required_fields)}'
        }), 400
    
    url = data['url'].strip()
    model_prediction = int(data['model_prediction'])
    human_label = int(data['human_label'])
    confidence = data.get('confidence')
    metadata = data.get('metadata', {})
    
    # Validate prediction and label values
    if model_prediction not in [0, 1] or human_label not in [0, 1]:
        return jsonify({
            'success': False,
            'error': 'model_prediction and human_label must be 0 or 1'
        }), 400
    
    try:
        feedback = feedback_collector.add_feedback(
            url_or_domain=url,
            model_prediction=model_prediction,
            human_label=human_label,
            confidence=float(confidence) if confidence is not None else None,
            feedback_type='correction' if model_prediction != human_label else 'confirmation',
            metadata=metadata
        )
        
        return jsonify({
            'success': True,
            'feedback_id': len(feedback_collector.feedback_history) - 1,
            'message': 'Feedback submitted successfully'
        }), 201
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to submit feedback: {str(e)}'
        }), 500


@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    """
    Retrieve all feedback entries.
    
    Query parameters (optional):
        - limit: Maximum number of entries to return
        - corrections_only: If true, only return corrections (default: false)
    
    Returns:
        JSON response with feedback entries
    """
    limit = request.args.get('limit', type=int)
    corrections_only = request.args.get('corrections_only', 'false').lower() == 'true'
    
    if corrections_only:
        feedback_list = feedback_collector.get_corrections()
    else:
        feedback_list = feedback_collector.feedback_history
    
    # Sort by timestamp (newest first)
    feedback_list = sorted(feedback_list, key=lambda x: x.timestamp or datetime.min, reverse=True)
    
    # Apply limit
    if limit:
        feedback_list = feedback_list[:limit]
    
    # Convert to dictionaries
    feedback_dicts = [f.to_dict() for f in feedback_list]
    
    return jsonify({
        'success': True,
        'feedback': feedback_dicts,
        'count': len(feedback_dicts),
        'total_count': feedback_collector.get_feedback_count(),
        'corrections_count': len(feedback_collector.get_corrections())
    })


@app.route('/api/feedback/count', methods=['GET'])
def get_feedback_count():
    """
    Get count of feedback entries.
    
    Returns:
        JSON response with feedback counts
    """
    return jsonify({
        'success': True,
        'total_count': feedback_collector.get_feedback_count(),
        'corrections_count': len(feedback_collector.get_corrections())
    })


@app.route('/api/train', methods=['POST'])
def trigger_training():
    """
    Trigger training execution.
    
    Request body (optional):
        - epochs: Number of training epochs (default: 50)
        - batch_size: Batch size (default: 32)
        - max_samples: Maximum samples per class (default: None)
    
    Returns:
        JSON response with training status
    """
    data = request.get_json() or {}
    
    # Check if training is already running
    with training_lock:
        if training_status['status'] == 'running':
            return jsonify({
                'success': False,
                'error': 'Training is already in progress'
            }), 409
    
    # Get parameters
    epochs = data.get('epochs', 50)
    batch_size = data.get('batch_size', 32)
    max_samples = data.get('max_samples')
    
    # Validate parameters
    if epochs < 1 or batch_size < 1:
        return jsonify({
            'success': False,
            'error': 'epochs and batch_size must be positive integers'
        }), 400
    
    # Start training in background thread
    thread = threading.Thread(
        target=run_training_thread,
        args=(epochs, batch_size, max_samples)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training started',
        'status': 'running'
    }), 202


@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """
    Get current training status.
    
    Returns:
        JSON response with training status
    """
    with training_lock:
        status_data = {
            'status': training_status['status'],
            'has_results': training_status['results'] is not None,
            'has_output': bool(training_status['output'])
        }
        
        if training_status['error']:
            status_data['error'] = training_status['error']
        
        return jsonify({
            'success': True,
            **status_data
        })


@app.route('/api/train/results', methods=['GET'])
def get_training_results():
    """
    Retrieve training results.
    
    Returns:
        JSON response with training results
    """
    with training_lock:
        if training_status['status'] == 'idle':
            return jsonify({
                'success': False,
                'error': 'No training has been run yet'
            }), 404
        
        if training_status['status'] == 'running':
            return jsonify({
                'success': False,
                'error': 'Training is still in progress'
            }), 409
        
        results = {
            'status': training_status['status'],
            'results': training_status['results'],
            'output': training_status['output']
        }
        
        if training_status['error']:
            results['error'] = training_status['error']
        
        return jsonify({
            'success': True,
            **results
        })


@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    """
    Get model status (whether model is loaded).
    
    Returns:
        JSON response with model status
    """
    model_instance = load_model_instance()
    
    return jsonify({
        'success': True,
        'loaded': model_instance is not None,
        'model_path': MODEL_PATH if model_instance is not None else None
    })


if __name__ == '__main__':
    # Load whitelist on startup
    load_whitelist_from_file()
    
    # Initialize feedback collector (already done at module level)
    print(f"Feedback collector initialized with {feedback_collector.get_feedback_count()} entries")
    
    # Run Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)
