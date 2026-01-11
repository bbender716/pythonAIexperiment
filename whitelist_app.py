"""
Flask web application for managing whitelist dataset.

This application provides a REST API for managing whitelist entries used
in training the ad-blocker AI model. It integrates with the AdBlockListParser
class to manage whitelist domains and persists them to a plain text file.
"""

import os
import urllib.parse
import re
from flask import Flask, request, jsonify, render_template
from typing import Optional, Set
from adblocker_ai import AdBlockListParser

app = Flask(__name__)

# Configuration
WHITELIST_FILE = 'whitelist_dataset.txt'

# Initialize parser
parser = AdBlockListParser()


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


if __name__ == '__main__':
    # Load whitelist on startup
    load_whitelist_from_file()
    
    # Run Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)
