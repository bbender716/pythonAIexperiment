"""
TensorFlow Ad-Blocker AI Model

This module implements a binary classifier that learns from Adblock Plus format
lists to identify ads in URLs and domains.
"""

import tensorflow as tf
import numpy as np
import requests
import urllib.parse
import re
from typing import List, Tuple, Set, Optional


class AdBlockListParser:
    """
    Parser for Adblock Plus format filter lists.
    
    Downloads and parses Adblock Plus format lists to extract blocked domains
    and URL patterns. Generates labeled samples for training.
    """
    
    def __init__(self):
        """Initialize the parser with default list URLs."""
        self.default_lists = [
            'https://easylist.to/easylist/easylist.txt',
            'https://easylist.to/easylist/easyprivacy.txt'
        ]
        self.blocked_domains: Set[str] = set()
        self.url_patterns: List[str] = []
        self.regex_patterns: List[str] = []
        self.domain_patterns: List[str] = []
        
    def download_list(self, url: str, timeout: int = 30) -> Optional[str]:
        """
        Download an Adblock Plus format list from a URL.
        
        Args:
            url: URL to download the list from
            timeout: Request timeout in seconds
            
        Returns:
            Content of the list as a string, or None if download fails
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error downloading list from {url}: {e}")
            return None
    
    def parse_line(self, line: str) -> Optional[dict]:
        """
        Parse a single line from an Adblock Plus format list.
        
        Args:
            line: A line from the filter list
            
        Returns:
            Dictionary with rule type and extracted information, or None if line should be skipped
        """
        # Remove whitespace
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('!') or line.startswith('[') or line.startswith('#'):
            return None
        
        # Skip exception rules (they allow content, not block it)
        if line.startswith('@@'):
            return None
        
        # Parse domain rules: ||domain.com^ or ||domain.com
        if line.startswith('||'):
            # Extract domain pattern
            # Format: ||domain.com^ or ||domain.com$options
            match = re.match(r'\|\|([^\^$\/]+)', line)
            if match:
                domain = match.group(1).strip()
                # Remove common options like $script, $image, etc.
                domain = re.sub(r'\$.*$', '', domain)
                if domain and not domain.startswith('!'):
                    return {
                        'type': 'domain',
                        'pattern': domain,
                        'original': line
                    }
        
        # Parse URL patterns that start with /
        if line.startswith('/') and line.endswith('/'):
            # Regex pattern: /pattern/
            pattern = line[1:-1]  # Remove leading and trailing /
            if pattern:
                return {
                    'type': 'regex',
                    'pattern': pattern,
                    'original': line
                }
        
        # Parse URL path patterns: /ads/banner.jpg or */ads/*
        if '/' in line and not line.startswith('!'):
            # Skip if it looks like an element hiding rule (contains ##)
            if '##' in line:
                return None
            
            # Extract URL pattern, handling wildcards
            # Remove domain-specific prefixes like ||domain.com
            pattern = re.sub(r'^\|\|[^\^$\/]+\^?', '', line)
            # Remove options like $image, $script, etc.
            pattern = re.sub(r'\$[^,]+(,.*)?$', '', pattern)
            
            if pattern and pattern != line:  # Only if we extracted something meaningful
                return {
                    'type': 'url_pattern',
                    'pattern': pattern,
                    'original': line
                }
            elif pattern and '/' in pattern:
                return {
                    'type': 'url_pattern',
                    'pattern': pattern,
                    'original': line
                }
        
        # Parse other domain patterns (without || prefix)
        # Format: domain.com or *.domain.com
        if re.match(r'^[\w\.\-\*]+$', line) and '.' in line and not line.startswith('/'):
            # Skip if it's an element hiding rule
            if '##' in line:
                return None
            # Skip if it has options (contains $)
            if '$' in line:
                return None
            
            domain = line.split('##')[0].strip()  # Remove element hiding part if present
            if domain and not domain.startswith('!'):
                return {
                    'type': 'domain_pattern',
                    'pattern': domain,
                    'original': line
                }
        
        return None
    
    def parse_list(self, content: str) -> dict:
        """
        Parse an entire Adblock Plus format list.
        
        Args:
            content: The full content of the filter list
            
        Returns:
            Dictionary containing parsed domains, URL patterns, and regex patterns
        """
        domains = set()
        url_patterns = []
        regex_patterns = []
        domain_patterns = []
        
        for line in content.split('\n'):
            parsed = self.parse_line(line)
            if parsed:
                if parsed['type'] == 'domain':
                    domains.add(parsed['pattern'])
                elif parsed['type'] == 'url_pattern':
                    url_patterns.append(parsed['pattern'])
                elif parsed['type'] == 'regex':
                    regex_patterns.append(parsed['pattern'])
                elif parsed['type'] == 'domain_pattern':
                    domain_patterns.append(parsed['pattern'])
        
        return {
            'domains': domains,
            'url_patterns': url_patterns,
            'regex_patterns': regex_patterns,
            'domain_patterns': domain_patterns
        }
    
    def download_and_parse(self, urls: Optional[List[str]] = None) -> dict:
        """
        Download and parse one or more Adblock Plus format lists.
        
        Args:
            urls: List of URLs to download. If None, uses default lists.
            
        Returns:
            Dictionary containing all parsed domains and patterns
        """
        if urls is None:
            urls = self.default_lists
        
        all_domains = set()
        all_url_patterns = []
        all_regex_patterns = []
        all_domain_patterns = []
        
        for url in urls:
            print(f"Downloading and parsing: {url}")
            content = self.download_list(url)
            if content:
                parsed = self.parse_list(content)
                all_domains.update(parsed['domains'])
                all_url_patterns.extend(parsed['url_patterns'])
                all_regex_patterns.extend(parsed['regex_patterns'])
                all_domain_patterns.extend(parsed['domain_patterns'])
                print(f"  Extracted {len(parsed['domains'])} domains, "
                      f"{len(parsed['url_patterns'])} URL patterns, "
                      f"{len(parsed['regex_patterns'])} regex patterns, "
                      f"{len(parsed['domain_patterns'])} domain patterns")
        
        self.blocked_domains = all_domains
        self.url_patterns = all_url_patterns
        self.regex_patterns = all_regex_patterns
        self.domain_patterns = all_domain_patterns
        
        return {
            'domains': all_domains,
            'url_patterns': all_url_patterns,
            'regex_patterns': all_regex_patterns,
            'domain_patterns': all_domain_patterns
        }
    
    def generate_positive_samples(self, max_samples: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Generate positive samples (ads) from parsed blocked domains and patterns.
        
        Args:
            max_samples: Maximum number of samples to generate. If None, generates all.
            
        Returns:
            List of tuples (url_or_domain, label) where label is 1 for ads
        """
        samples = []
        
        # Add domains as samples
        domain_list = list(self.blocked_domains)
        if max_samples:
            domain_list = domain_list[:max_samples]
        
        for domain in domain_list:
            samples.append((domain, 1))
            # Also add with common protocols
            samples.append((f"http://{domain}", 1))
            samples.append((f"https://{domain}", 1))
        
        # Add domain patterns
        if max_samples:
            remaining = max_samples - len(samples)
            if remaining > 0:
                domain_pattern_list = list(self.domain_patterns)[:remaining]
                for pattern in domain_pattern_list:
                    samples.append((pattern, 1))
        else:
            for pattern in self.domain_patterns:
                samples.append((pattern, 1))
        
        # Generate URLs from URL patterns (simplified - just use the pattern as-is)
        if max_samples:
            remaining = max_samples - len(samples)
            if remaining > 0:
                url_pattern_list = self.url_patterns[:remaining]
                for pattern in url_pattern_list:
                    # Create a sample URL using a placeholder domain
                    samples.append((f"https://example.com{pattern}", 1))
        else:
            for pattern in self.url_patterns:
                samples.append((f"https://example.com{pattern}", 1))
        
        return samples
    
    def get_blocked_domains(self) -> Set[str]:
        """Return the set of blocked domains."""
        return self.blocked_domains.copy()
    
    def get_url_patterns(self) -> List[str]:
        """Return the list of URL patterns."""
        return self.url_patterns.copy()
    
    def get_regex_patterns(self) -> List[str]:
        """Return the list of regex patterns."""
        return self.regex_patterns.copy()
    
    def get_domain_patterns(self) -> List[str]:
        """Return the list of domain patterns."""
        return self.domain_patterns.copy()


def generate_dataset(
    parser: AdBlockListParser,
    max_samples_per_class: Optional[int] = None,
    balance_dataset: bool = True,
    seed: Optional[int] = None
) -> List[Tuple[str, int]]:
    """
    Generate labeled dataset from parsed Adblock Plus lists.
    
    Creates a balanced dataset with positive samples (ads=1) from blocked patterns
    and negative samples (legitimate=0) from common legitimate domains.
    
    Args:
        parser: AdBlockListParser instance with parsed lists
        max_samples_per_class: Maximum number of samples per class (ads/legitimate).
                              If None, uses all available samples.
        balance_dataset: If True, ensures equal number of positive and negative samples.
                        If False, uses all available samples.
        seed: Random seed for reproducibility (for shuffling and sampling)
        
    Returns:
        List of tuples (url_or_domain, label) where label is 1 for ads, 0 for legitimate
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    # Common legitimate domains for negative samples
    # These are well-known legitimate websites that should not be blocked
    LEGITIMATE_DOMAINS = [
        # Major platforms
        'google.com', 'github.com', 'stackoverflow.com', 'wikipedia.org',
        'youtube.com', 'reddit.com', 'twitter.com', 'facebook.com',
        'microsoft.com', 'apple.com', 'amazon.com', 'netflix.com',
        # News and media
        'bbc.com', 'cnn.com', 'nytimes.com', 'reuters.com', 'theguardian.com',
        # Educational
        'edu', 'mit.edu', 'stanford.edu', 'harvard.edu', 'coursera.org',
        # Technology
        'mozilla.org', 'python.org', 'numpy.org', 'pytorch.org',
        'tensorflow.org', 'keras.io', 'jupyter.org',
        # General legitimate patterns
        'example.com', 'example.org', 'example.net',
        # Government
        'gov', 'usa.gov', 'europa.eu',
    ]
    
    # Generate positive samples (ads) from parsed lists
    positive_samples = parser.generate_positive_samples(max_samples=max_samples_per_class)
    
    # Generate negative samples (legitimate)
    negative_samples = []
    
    # Use legitimate domains
    legitimate_list = LEGITIMATE_DOMAINS.copy()
    if max_samples_per_class:
        # If we need to limit samples, take a subset
        if len(legitimate_list) > max_samples_per_class:
            legitimate_list = random.sample(legitimate_list, max_samples_per_class)
    
    for domain in legitimate_list:
        # Add domain as-is
        negative_samples.append((domain, 0))
        # Add with protocols
        negative_samples.append((f"http://{domain}", 0))
        negative_samples.append((f"https://{domain}", 0))
        
        # Add with common paths (simulating legitimate URLs)
        common_paths = ['/', '/index.html', '/about', '/contact', '/help', 
                       '/docs', '/api', '/blog', '/news', '/products']
        for path in common_paths[:3]:  # Limit to first 3 paths per domain
            negative_samples.append((f"https://{domain}{path}", 0))
    
    # If we still need more negative samples and max_samples_per_class is set,
    # generate more from legitimate domain patterns
    if max_samples_per_class and len(negative_samples) < max_samples_per_class:
        # Generate additional legitimate domain patterns
        legitimate_tlds = ['com', 'org', 'net', 'edu', 'gov', 'io', 'co']
        legitimate_names = ['company', 'business', 'services', 'support', 'info',
                          'main', 'www', 'shop', 'store', 'blog', 'news', 'help']
        
        remaining = max_samples_per_class - len(negative_samples)
        for i in range(min(remaining, 50)):  # Limit additional generation
            tld = random.choice(legitimate_tlds)
            name = random.choice(legitimate_names)
            domain = f"{name}.{tld}"
            negative_samples.append((domain, 0))
            negative_samples.append((f"https://{domain}", 0))
    
    # Limit negative samples if max_samples_per_class is set
    if max_samples_per_class and len(negative_samples) > max_samples_per_class:
        negative_samples = random.sample(negative_samples, max_samples_per_class)
    
    # Balance dataset if requested
    if balance_dataset:
        min_samples = min(len(positive_samples), len(negative_samples))
        # Sample equal number from each class (only if we have samples)
        if min_samples > 0:
            positive_samples = random.sample(positive_samples, min_samples)
            negative_samples = random.sample(negative_samples, min_samples)
        else:
            # If either class is empty, return empty dataset
            return []
    
    # Combine and shuffle
    dataset = positive_samples + negative_samples
    random.shuffle(dataset)
    
    return dataset


def prepare_training_data(
    dataset: List[Tuple[str, int]],
    feature_extractor: Optional['FeatureExtractor'] = None,
    use_url_features: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    normalize: bool = True,
    stratify: bool = True,
    seed: Optional[int] = None,
    verbose: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Prepare training data with preprocessing, normalization, and train/val/test split.
    
    This function implements a complete training pipeline:
    1. Extract features from raw dataset (URLs/domains)
    2. Preprocess and normalize features (fit on training set, apply to all splits)
    3. Split into train/validation/test sets with optional stratification
    4. Return all splits ready for model training
    
    Args:
        dataset: List of tuples (url_or_domain, label) where label is 0 or 1
        feature_extractor: FeatureExtractor instance. If None, creates a new one.
        use_url_features: If True, extracts full URL features (25 dim). If False, uses domain-only features (8 dim).
        train_ratio: Proportion of data for training (default 0.7)
        val_ratio: Proportion of data for validation (default 0.15)
        test_ratio: Proportion of data for testing (default 0.15)
        normalize: If True, applies normalization fitted on training data (default True)
        stratify: If True, uses stratified splitting to maintain class balance (default True)
        seed: Random seed for reproducibility (for shuffling and splitting)
        verbose: Verbosity level (0=silent, 1=info messages)
        
    Returns:
        Tuple containing:
        - X_train: Training features (numpy array)
        - y_train: Training labels (numpy array)
        - X_val: Validation features (numpy array)
        - y_val: Validation labels (numpy array)
        - X_test: Test features (numpy array)
        - y_test: Test labels (numpy array)
        - normalization_stats: Dictionary with normalization parameters (mean, std, min, max per feature)
        
    Raises:
        ValueError: If ratios don't sum to 1.0, or dataset is empty
    """
    import random
    from collections import Counter
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    if not dataset:
        raise ValueError("Dataset is empty. Cannot prepare training data.")
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize feature extractor if not provided
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
    
    if verbose >= 1:
        print(f"Preparing training data from {len(dataset)} samples...")
    
    # Step 1: Extract features from all samples
    features_list = []
    labels_list = []
    
    if verbose >= 1:
        print("Extracting features from dataset...")
    
    for url_or_domain, label in dataset:
        # Determine which feature extraction method to use
        if use_url_features:
            # Use URL features (25 dimensions)
            if '/' in url_or_domain or url_or_domain.startswith('http'):
                features = feature_extractor.extract_url_features(url_or_domain)
            else:
                # Domain only, but we want URL features
                # Pad domain features or extract as URL (extract_url_features handles this)
                features = feature_extractor.extract_url_features(url_or_domain)
        else:
            # Use domain-only features (8 dimensions)
            features = feature_extractor.extract_domain_features(url_or_domain)
        
        features_list.append(features)
        labels_list.append(label)
    
    # Convert to numpy arrays
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32)
    
    feature_dim = X.shape[1]
    if verbose >= 1:
        print(f"  Extracted {feature_dim}-dimensional features from {len(X)} samples")
        label_counts = Counter(y)
        print(f"  Class distribution: {dict(label_counts)}")
    
    # Step 2: Split into train/val/test sets
    n_samples = len(X)
    
    if stratify:
        # Stratified split to maintain class balance
        # Separate samples by class
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]
        
        # Shuffle each class separately
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(class_0_indices)
        np.random.shuffle(class_1_indices)
        
        # Calculate split sizes for each class
        n_train_0 = int(len(class_0_indices) * train_ratio)
        n_val_0 = int(len(class_0_indices) * val_ratio)
        n_train_1 = int(len(class_1_indices) * train_ratio)
        n_val_1 = int(len(class_1_indices) * val_ratio)
        
        # Split indices for each class
        train_indices_0 = class_0_indices[:n_train_0]
        val_indices_0 = class_0_indices[n_train_0:n_train_0 + n_val_0]
        test_indices_0 = class_0_indices[n_train_0 + n_val_0:]
        
        train_indices_1 = class_1_indices[:n_train_1]
        val_indices_1 = class_1_indices[n_train_1:n_train_1 + n_val_1]
        test_indices_1 = class_1_indices[n_train_1 + n_val_1:]
        
        # Combine indices
        train_indices = np.concatenate([train_indices_0, train_indices_1])
        val_indices = np.concatenate([val_indices_0, val_indices_1])
        test_indices = np.concatenate([test_indices_0, test_indices_1])
        
        # Shuffle combined indices to mix classes
        if seed is not None:
            np.random.seed(seed + 1)  # Different seed for final shuffle
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        # Split data
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
    else:
        # Simple random split
        # Shuffle indices
        indices = np.arange(n_samples)
        random.shuffle(indices)
        
        # Calculate split points
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Split data
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
    
    if verbose >= 1:
        print(f"\nData split:")
        print(f"  Training set:   {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
        print(f"  Test set:       {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
        print(f"  Training class distribution: {dict(Counter(y_train))}")
        print(f"  Validation class distribution: {dict(Counter(y_val))}")
        print(f"  Test class distribution: {dict(Counter(y_test))}")
    
    # Step 3: Normalization (fit on training set, apply to all splits)
    normalization_stats = {}
    
    if normalize:
        if verbose >= 1:
            print("\nNormalizing features (fit on training set)...")
        
        # Compute statistics from training set
        feature_mean = np.mean(X_train, axis=0, keepdims=True)
        feature_std = np.std(X_train, axis=0, keepdims=True)
        feature_min = np.min(X_train, axis=0, keepdims=True)
        feature_max = np.max(X_train, axis=0, keepdims=True)
        
        # Avoid division by zero for constant features
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
        
        # Store normalization statistics
        normalization_stats = {
            'mean': feature_mean.flatten().copy(),
            'std': feature_std.flatten().copy(),
            'min': feature_min.flatten().copy(),
            'max': feature_max.flatten().copy(),
            'normalized': True
        }
        
        # Apply standardization (z-score normalization): (x - mean) / std
        # This centers features at 0 and scales them to unit variance
        X_train_norm = (X_train - feature_mean) / feature_std
        X_val_norm = (X_val - feature_mean) / feature_std
        X_test_norm = (X_test - feature_mean) / feature_std
        
        # Optional: Also clip outliers to reasonable range (e.g., [-3, 3] standard deviations)
        # This helps with stability during training
        clip_range = 5.0  # Clip at 5 standard deviations
        X_train_norm = np.clip(X_train_norm, -clip_range, clip_range)
        X_val_norm = np.clip(X_val_norm, -clip_range, clip_range)
        X_test_norm = np.clip(X_test_norm, -clip_range, clip_range)
        
        if verbose >= 1:
            print(f"  Applied standardization: mean=0, std=1 (clipped at ±{clip_range} std)")
            print(f"  Feature ranges after normalization:")
            print(f"    Training:   [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
            print(f"    Validation: [{X_val_norm.min():.3f}, {X_val_norm.max():.3f}]")
            print(f"    Test:       [{X_test_norm.min():.3f}, {X_test_norm.max():.3f}]")
        
        X_train = X_train_norm
        X_val = X_val_norm
        X_test = X_test_norm
    else:
        normalization_stats = {'normalized': False}
        if verbose >= 1:
            print("\nSkipping normalization (normalize=False)")
    
    if verbose >= 1:
        print("\nTraining data preparation complete!")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, normalization_stats


class FeatureExtractor:
    """
    Feature extraction from URLs and domains.
    
    Extracts numerical features from URLs and domains that can be used for
    machine learning classification. Features include domain characteristics,
    path patterns, query parameters, and ad-related keyword patterns.
    """
    
    # Common ad-related keywords in URLs/paths
    AD_KEYWORDS = [
        'ad', 'ads', 'advert', 'advertising', 'banner', 'popup', 'pop-under',
        'sponsor', 'promo', 'promotion', 'affiliate', 'track', 'tracking',
        'analytics', 'beacon', 'click', 'clicks', 'delivery', 'doubleclick',
        'googleadservices', 'googlesyndication', 'outbrain', 'taboola',
        'amazon-adsystem', 'adsystem', 'adnxs', 'advertising', 'media',
        'marketing', 'campaign', 'serve', 'server', 'delivery'
    ]
    
    # Common tracking parameters in query strings
    TRACKING_PARAMS = [
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'ref', 'referrer', 'referer', 'source', 'campaign', 'click_id',
        'affiliate_id', 'partner_id', 'track', 'tracking', 'sid', 'session_id',
        'user_id', 'uid', 'cid', 'gclid', 'fbclid', 'dclid', 'twclid'
    ]
    
    # Known ad network domain patterns
    AD_NETWORK_PATTERNS = [
        'doubleclick', 'googlesyndication', 'googleadservices', 'adnxs',
        'advertising', 'adsystem', 'amazon-adsystem', 'outbrain', 'taboola',
        'criteo', 'adtech', 'adtechus', 'rubiconproject', 'openx', 'pubmatic',
        'indexexchange', 'appnexus', 'adsrvr', 'adserver', 'adform', 'adsafeprotected'
    ]
    
    # Common ad-related file extensions
    AD_FILE_EXTENSIONS = ['js', 'gif', 'jpg', 'jpeg', 'png', 'swf', 'xml', 'json']
    
    # Common TLDs (top-level domains)
    COMMON_TLDS = ['com', 'org', 'net', 'edu', 'gov', 'co', 'io', 'me', 'tv', 'info']
    
    def __init__(self):
        """Initialize the feature extractor."""
        # Compile regex patterns for efficiency
        self.ad_keyword_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.AD_KEYWORDS),
            re.IGNORECASE
        )
        self.ad_network_pattern = re.compile(
            '|'.join(re.escape(patt) for patt in self.AD_NETWORK_PATTERNS),
            re.IGNORECASE
        )
    
    def _extract_domain_info(self, domain: str) -> dict:
        """
        Extract domain-level information.
        
        Args:
            domain: Domain string (e.g., "sub.example.com")
            
        Returns:
            Dictionary with domain characteristics
        """
        if not domain:
            return {
                'domain_length': 0,
                'subdomain_count': 0,
                'has_www': False,
                'tld_length': 0,
                'is_common_tld': False,
                'has_numeric': False,
                'has_hyphen': False,
                'contains_ad_network': False
            }
        
        # Remove protocol if present
        domain = domain.replace('http://', '').replace('https://', '').split('/')[0]
        domain = domain.split(':')[0]  # Remove port
        
        parts = domain.split('.')
        
        # Domain length
        domain_length = len(domain)
        
        # Subdomain count (everything except TLD and main domain)
        subdomain_count = max(0, len(parts) - 2)
        
        # Has www
        has_www = domain.lower().startswith('www.')
        
        # TLD characteristics
        tld = parts[-1] if len(parts) > 0 else ''
        tld_length = len(tld)
        is_common_tld = tld.lower() in self.COMMON_TLDS
        
        # Domain structure features
        has_numeric = bool(re.search(r'\d', domain))
        has_hyphen = '-' in domain
        
        # Check for ad network patterns
        contains_ad_network = bool(self.ad_network_pattern.search(domain))
        
        return {
            'domain_length': domain_length,
            'subdomain_count': subdomain_count,
            'has_www': has_www,
            'tld_length': tld_length,
            'is_common_tld': is_common_tld,
            'has_numeric': has_numeric,
            'has_hyphen': has_hyphen,
            'contains_ad_network': contains_ad_network
        }
    
    def _extract_path_features(self, path: str) -> dict:
        """
        Extract features from URL path.
        
        Args:
            path: URL path string (e.g., "/ads/banner.jpg")
            
        Returns:
            Dictionary with path characteristics
        """
        if not path:
            return {
                'path_length': 0,
                'path_depth': 0,
                'contains_ad_keyword': False,
                'ad_keyword_count': 0,
                'has_file_extension': False,
                'is_image_file': False,
                'has_numeric': False,
                'has_uuid_pattern': False
            }
        
        # Path length
        path_length = len(path)
        
        # Path depth (number of segments)
        path_segments = [seg for seg in path.split('/') if seg]
        path_depth = len(path_segments)
        
        # Check for ad keywords
        ad_keyword_matches = self.ad_keyword_pattern.findall(path.lower())
        contains_ad_keyword = len(ad_keyword_matches) > 0
        ad_keyword_count = len(ad_keyword_matches)
        
        # File extension features
        has_file_extension = '.' in path
        is_image_file = any(path.lower().endswith(f'.{ext}') for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg'])
        
        # Numeric patterns
        has_numeric = bool(re.search(r'\d', path))
        
        # UUID pattern (common in tracking URLs)
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        has_uuid_pattern = bool(re.search(uuid_pattern, path.lower()))
        
        return {
            'path_length': path_length,
            'path_depth': path_depth,
            'contains_ad_keyword': contains_ad_keyword,
            'ad_keyword_count': ad_keyword_count,
            'has_file_extension': has_file_extension,
            'is_image_file': is_image_file,
            'has_numeric': has_numeric,
            'has_uuid_pattern': has_uuid_pattern
        }
    
    def _extract_query_features(self, query_string: str) -> dict:
        """
        Extract features from URL query string.
        
        Args:
            query_string: URL query string (e.g., "utm_source=google&id=123")
            
        Returns:
            Dictionary with query characteristics
        """
        if not query_string:
            return {
                'query_length': 0,
                'param_count': 0,
                'has_tracking_params': False,
                'tracking_param_count': 0,
                'has_numeric_values': False,
                'has_uuid_in_query': False
            }
        
        # Query length
        query_length = len(query_string)
        
        # Parse query parameters
        params = urllib.parse.parse_qs(query_string)
        param_count = len(params)
        
        # Check for tracking parameters
        tracking_params = [k for k in params.keys() if any(tp in k.lower() for tp in self.TRACKING_PARAMS)]
        has_tracking_params = len(tracking_params) > 0
        tracking_param_count = len(tracking_params)
        
        # Check for numeric values
        has_numeric_values = bool(re.search(r'=\d+', query_string))
        
        # Check for UUID in query
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        has_uuid_in_query = bool(re.search(uuid_pattern, query_string.lower()))
        
        return {
            'query_length': query_length,
            'param_count': param_count,
            'has_tracking_params': has_tracking_params,
            'tracking_param_count': tracking_param_count,
            'has_numeric_values': has_numeric_values,
            'has_uuid_in_query': has_uuid_in_query
        }
    
    def extract_url_features(self, url: str) -> np.ndarray:
        """
        Extract features from a full URL.
        
        Args:
            url: Full URL string (e.g., "https://example.com/ads/banner.jpg?tracking=123")
            
        Returns:
            Normalized feature vector as numpy array
        """
        if not url:
            # Return zero vector with same dimension as normal features
            return np.zeros(25, dtype=np.float32)
        
        # Parse URL
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            # If URL parsing fails, treat as empty
            return np.zeros(25, dtype=np.float32)
        
        # Extract components
        domain = parsed.netloc or parsed.hostname or ''
        path = parsed.path or ''
        query = parsed.query or ''
        
        # Extract domain features
        domain_info = self._extract_domain_info(domain)
        
        # Extract path features
        path_info = self._extract_path_features(path)
        
        # Extract query features
        query_info = self._extract_query_features(query)
        
        # Combine all features into a single vector
        features = [
            # Domain features (8)
            domain_info['domain_length'],
            domain_info['subdomain_count'],
            float(domain_info['has_www']),
            domain_info['tld_length'],
            float(domain_info['is_common_tld']),
            float(domain_info['has_numeric']),
            float(domain_info['has_hyphen']),
            float(domain_info['contains_ad_network']),
            
            # Path features (8)
            path_info['path_length'],
            path_info['path_depth'],
            float(path_info['contains_ad_keyword']),
            path_info['ad_keyword_count'],
            float(path_info['has_file_extension']),
            float(path_info['is_image_file']),
            float(path_info['has_numeric']),
            float(path_info['has_uuid_pattern']),
            
            # Query features (6)
            query_info['query_length'],
            query_info['param_count'],
            float(query_info['has_tracking_params']),
            query_info['tracking_param_count'],
            float(query_info['has_numeric_values']),
            float(query_info['has_uuid_in_query']),
            
            # URL-level features (3)
            len(url),  # Total URL length
            float(self.ad_keyword_pattern.search(url.lower()) is not None),  # Contains ad keyword anywhere
            float(self.ad_network_pattern.search(url.lower()) is not None)  # Contains ad network anywhere
        ]
        
        # Convert to numpy array and normalize
        feature_vector = np.array(features, dtype=np.float32)
        
        # Normalize features to [0, 1] range using min-max scaling
        # For features that are counts/lengths, apply log scaling + normalization
        # For boolean features, they're already 0 or 1
        
        # Normalize length/count features with log scaling for better distribution
        length_indices = [0, 1, 3, 8, 9, 11, 17, 18, 20, 23]  # Length and count features
        for idx in length_indices:
            if feature_vector[idx] > 0:
                feature_vector[idx] = np.log1p(feature_vector[idx])  # log(1 + x)
        
        # Apply min-max normalization (using reasonable max values)
        max_values = np.array([
            200.0,  # domain_length (log scaled)
            5.0,    # subdomain_count (log scaled)
            1.0,    # has_www
            10.0,   # tld_length (log scaled)
            1.0,    # is_common_tld
            1.0,    # has_numeric
            1.0,    # has_hyphen
            1.0,    # contains_ad_network
            200.0,  # path_length (log scaled)
            10.0,   # path_depth (log scaled)
            1.0,    # contains_ad_keyword
            10.0,   # ad_keyword_count (log scaled)
            1.0,    # has_file_extension
            1.0,    # is_image_file
            1.0,    # has_numeric (path)
            1.0,    # has_uuid_pattern
            200.0,  # query_length (log scaled)
            20.0,   # param_count (log scaled)
            1.0,    # has_tracking_params
            10.0,   # tracking_param_count (log scaled)
            1.0,    # has_numeric_values
            1.0,    # has_uuid_in_query
            500.0,  # total_url_length (log scaled)
            1.0,    # contains_ad_keyword (url)
            1.0     # contains_ad_network (url)
        ], dtype=np.float32)
        
        # Normalize
        feature_vector = np.clip(feature_vector / max_values, 0.0, 1.0)
        
        return feature_vector
    
    def extract_domain_features(self, domain: str) -> np.ndarray:
        """
        Extract features from a domain only (no URL path).
        
        Args:
            domain: Domain string (e.g., "ads.example.com" or "http://ads.example.com")
            
        Returns:
            Normalized feature vector as numpy array
        """
        if not domain:
            # Return zero vector with same dimension as domain-only features
            return np.zeros(8, dtype=np.float32)
        
        # Extract domain info
        domain_info = self._extract_domain_info(domain)
        
        # Combine domain features into a vector
        features = [
            # Domain features (8)
            domain_info['domain_length'],
            domain_info['subdomain_count'],
            float(domain_info['has_www']),
            domain_info['tld_length'],
            float(domain_info['is_common_tld']),
            float(domain_info['has_numeric']),
            float(domain_info['has_hyphen']),
            float(domain_info['contains_ad_network'])
        ]
        
        # Convert to numpy array
        feature_vector = np.array(features, dtype=np.float32)
        
        # Normalize features
        max_values = np.array([
            200.0,  # domain_length (log scaled)
            5.0,    # subdomain_count (log scaled)
            1.0,    # has_www
            10.0,   # tld_length (log scaled)
            1.0,    # is_common_tld
            1.0,    # has_numeric
            1.0,    # has_hyphen
            1.0     # contains_ad_network
        ], dtype=np.float32)
        
        # Normalize length/count features with log scaling
        length_indices = [0, 1, 3]
        for idx in length_indices:
            if feature_vector[idx] > 0:
                feature_vector[idx] = np.log1p(feature_vector[idx])
        
        # Apply min-max normalization
        feature_vector = np.clip(feature_vector / max_values, 0.0, 1.0)
        
        return feature_vector


class F1Score(tf.keras.metrics.Metric):
    """
    Custom F1-score metric for binary classification.
    
    F1-score is the harmonic mean of precision and recall:
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


class AdBlockerModel:
    """
    TensorFlow/Keras binary classifier for ad blocking.
    
    This class implements a neural network model that classifies URLs and domains
    as ads (1) or legitimate (0) based on extracted features.
    """
    
    def __init__(self, input_dim: int = 25, hidden_units: Optional[List[int]] = None):
        """
        Initialize the AdBlocker model.
        
        Args:
            input_dim: Number of input features (default 25 for URL features, 8 for domain-only)
            hidden_units: List of hidden layer sizes. If None, uses default [64, 32]
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units if hidden_units is not None else [64, 32]
        self.model: Optional[tf.keras.Model] = None
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
    
    def build_model(self) -> tf.keras.Model:
        """
        Build the TensorFlow/Keras model architecture.
        
        Architecture:
        - Input Layer: Dense layer with input_dim features
        - Hidden Layers: 2-3 dense layers with ReLU activation
        - Output Layer: Single neuron with sigmoid activation (binary classification)
        - Loss: Binary crossentropy
        - Optimizer: Adam
        - Metrics: Accuracy, Precision, Recall, F1-score
        
        Returns:
            Compiled Keras model
        """
        # Return existing model if already built
        if self.model is not None:
            return self.model
        
        model = tf.keras.Sequential()
        
        # Input layer (first hidden layer)
        model.add(tf.keras.layers.Dense(
            units=self.hidden_units[0],
            activation='relu',
            input_shape=(self.input_dim,),
            name='input_layer'
        ))
        
        # Add dropout for regularization
        model.add(tf.keras.layers.Dropout(0.2, name='dropout_0'))
        
        # Additional hidden layers
        for i, units in enumerate(self.hidden_units[1:], start=1):
            model.add(tf.keras.layers.Dense(
                units=units,
                activation='relu',
                name=f'hidden_layer_{i}'
            ))
            model.add(tf.keras.layers.Dropout(0.2, name=f'dropout_{i}'))
        
        # Output layer: single neuron with sigmoid activation
        model.add(tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            name='output_layer'
        ))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                F1Score(name='f1_score')
            ]
        )
        
        self.model = model
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X: Training features (numpy array of shape (n_samples, n_features))
            y: Training labels (numpy array of shape (n_samples,) with values 0 or 1)
            validation_data: Optional tuple (X_val, y_val) for validation set
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history object
        """
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Convert inputs to numpy arrays if needed
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Validate input shapes
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} != {y.shape[0]}")
        if X.shape[1] != self.input_dim:
            raise ValueError(f"X feature dimension {X.shape[1]} does not match model input_dim {self.input_dim}")
        
        # Set up callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        # Train the model
        history = self.model.fit(
            X,
            y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        
        self.is_trained = True
        return history
    
    def predict(self, url_or_domain: str, threshold: float = 0.5) -> Tuple[int, float]:
        """
        Predict whether a URL or domain is an ad.
        
        Args:
            url_or_domain: URL or domain string to classify
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Tuple of (prediction, confidence) where:
            - prediction: 0 (legitimate) or 1 (ad)
            - confidence: Model's confidence score (probability)
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() or load_model() first.")
        
        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Extract features
        # Try URL features first (25 features), fall back to domain features (8 features)
        if '/' in url_or_domain or url_or_domain.startswith('http'):
            features = self.feature_extractor.extract_url_features(url_or_domain)
            if self.input_dim == 8:
                # Model expects domain features, use domain-only extractor
                features = self.feature_extractor.extract_domain_features(url_or_domain)
        else:
            features = self.feature_extractor.extract_domain_features(url_or_domain)
            if self.input_dim == 25:
                # Model expects URL features, use URL extractor (will pad if needed)
                features = self.feature_extractor.extract_url_features(url_or_domain)
        
        # Ensure feature dimension matches model input
        if features.shape[0] != self.input_dim:
            raise ValueError(
                f"Feature dimension {features.shape[0]} does not match model input_dim {self.input_dim}. "
                f"Use appropriate FeatureExtractor method or adjust model input_dim."
            )
        
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Predict
        confidence = float(self.model.predict(features, verbose=0)[0][0])
        prediction = 1 if confidence >= threshold else 0
        
        return prediction, confidence
    
    def predict_batch(self, urls_or_domains: List[str], threshold: float = 0.5) -> List[Tuple[int, float]]:
        """
        Predict on a batch of URLs or domains.
        
        Args:
            urls_or_domains: List of URL or domain strings to classify
            threshold: Classification threshold (default 0.5)
            
        Returns:
            List of tuples (prediction, confidence) for each input
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() or load_model() first.")
        
        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Extract features for all samples
        features_list = []
        for url_or_domain in urls_or_domains:
            if '/' in url_or_domain or url_or_domain.startswith('http'):
                features = self.feature_extractor.extract_url_features(url_or_domain)
            else:
                features = self.feature_extractor.extract_domain_features(url_or_domain)
            
            # Handle dimension mismatch
            if features.shape[0] != self.input_dim:
                if self.input_dim == 25 and features.shape[0] == 8:
                    # Pad domain features to match URL features (not ideal, but workable)
                    features = np.pad(features, (0, 25 - 8), 'constant')
                elif self.input_dim == 8 and features.shape[0] == 25:
                    # Use first 8 features (domain features are first 8)
                    features = features[:8]
                else:
                    raise ValueError(
                        f"Feature dimension {features.shape[0]} does not match model input_dim {self.input_dim}"
                    )
            
            features_list.append(features)
        
        # Stack into batch
        X_batch = np.array(features_list, dtype=np.float32)
        
        # Predict
        confidences = self.model.predict(X_batch, verbose=0).flatten()
        predictions = [(1 if conf >= threshold else 0, float(conf)) for conf in confidences]
        
        return predictions
    
    def save_model(self, path: str):
        """
        Save the trained model to disk.
        
        Args:
            path: Directory path where to save the model (TensorFlow SavedModel format)
        """
        if self.model is None:
            raise ValueError("Model has not been built. Nothing to save.")
        
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a saved model from disk.
        
        Args:
            path: Directory path where the model is saved
        """
        try:
            self.model = tf.keras.models.load_model(path)
            # Infer input_dim from loaded model
            input_shape = self.model.input_shape
            if input_shape and len(input_shape) > 1:
                self.input_dim = int(input_shape[1])
            self.is_trained = True
            print(f"Model loaded from {path}")
        except Exception as e:
            raise ValueError(f"Failed to load model from {path}: {e}")
    
    def get_model_summary(self) -> str:
        """
        Get a string summary of the model architecture.
        
        Returns:
            Model summary as string
        """
        if self.model is None:
            return "Model has not been built yet."
        
        from io import StringIO
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: int = 1) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features (numpy array)
            y: Test labels (numpy array)
            verbose: Verbosity mode
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() or load_model() first.")
        
        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        results = self.model.evaluate(X, y, verbose=verbose, return_dict=True)
        return results


# ============================================================================
# RLHF/RLHP (Reinforcement Learning from Human Feedback/Preferences) Support
# ============================================================================

from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import json


@dataclass
class HumanFeedback:
    """
    Data structure for storing human feedback/preferences.
    
    Used for RLHF/RLHP (Reinforcement Learning from Human Feedback/Preferences)
    to collect and store human judgments about model predictions.
    
    Attributes:
        url_or_domain: The URL or domain that was evaluated
        model_prediction: The model's prediction (0=legitimate, 1=ad)
        human_label: The human's label/feedback (0=legitimate, 1=ad)
        confidence: Optional confidence score from the model
        feedback_type: Type of feedback ('correction', 'preference', 'ranking')
        timestamp: When the feedback was collected
        user_id: Optional identifier for the user providing feedback
        metadata: Optional additional metadata (dict)
    """
    url_or_domain: str
    model_prediction: int
    human_label: int
    confidence: Optional[float] = None
    feedback_type: str = 'correction'  # 'correction', 'preference', 'ranking'
    timestamp: Optional[datetime] = None
    user_id: Optional[str] = None
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> dict:
        """Convert feedback to dictionary for serialization."""
        return {
            'url_or_domain': self.url_or_domain,
            'model_prediction': self.model_prediction,
            'human_label': self.human_label,
            'confidence': self.confidence,
            'feedback_type': self.feedback_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'user_id': self.user_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HumanFeedback':
        """Create HumanFeedback from dictionary."""
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class PreferencePair:
    """
    Data structure for preference-based feedback (comparing two items).
    
    Used in RLHF/RLHP for ranking/comparison scenarios where humans
    indicate which of two options they prefer.
    
    Attributes:
        item_a: First URL/domain to compare
        item_b: Second URL/domain to compare
        preferred: Which item is preferred ('a' or 'b')
        context: Optional context for the comparison
        timestamp: When the preference was collected
        user_id: Optional identifier for the user
    """
    item_a: str
    item_b: str
    preferred: str  # 'a' or 'b'
    context: Optional[str] = None
    timestamp: Optional[datetime] = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamp and validate preferred."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.preferred not in ['a', 'b']:
            raise ValueError("preferred must be 'a' or 'b'")


class FeedbackCollector:
    """
    Collector for human feedback/preferences.
    
    Stores and manages human feedback data that can be used for
    RLHF/RLHP training. Provides methods to collect, store, and export feedback.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feedback collector.
        
        Args:
            storage_path: Optional path to JSON file for persistent storage
        """
        self.feedback_history: List[HumanFeedback] = []
        self.preference_pairs: List[PreferencePair] = []
        self.storage_path = storage_path
        
        # Load existing feedback if storage path exists
        if storage_path:
            self.load_from_file(storage_path)
    
    def add_feedback(
        self,
        url_or_domain: str,
        model_prediction: int,
        human_label: int,
        confidence: Optional[float] = None,
        feedback_type: str = 'correction',
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> HumanFeedback:
        """
        Add human feedback.
        
        Args:
            url_or_domain: URL or domain evaluated
            model_prediction: Model's prediction (0 or 1)
            human_label: Human's label (0 or 1)
            confidence: Model's confidence score
            feedback_type: Type of feedback
            user_id: User identifier
            metadata: Additional metadata
            
        Returns:
            Created HumanFeedback object
        """
        feedback = HumanFeedback(
            url_or_domain=url_or_domain,
            model_prediction=model_prediction,
            human_label=human_label,
            confidence=confidence,
            feedback_type=feedback_type,
            user_id=user_id,
            metadata=metadata
        )
        self.feedback_history.append(feedback)
        
        # Auto-save if storage path is set
        if self.storage_path:
            self.save_to_file(self.storage_path)
        
        return feedback
    
    def add_preference(
        self,
        item_a: str,
        item_b: str,
        preferred: str,
        context: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> PreferencePair:
        """
        Add preference pair (for ranking/comparison scenarios).
        
        Args:
            item_a: First URL/domain
            item_b: Second URL/domain
            preferred: Which is preferred ('a' or 'b')
            context: Optional context
            user_id: User identifier
            
        Returns:
            Created PreferencePair object
        """
        preference = PreferencePair(
            item_a=item_a,
            item_b=item_b,
            preferred=preferred,
            context=context,
            user_id=user_id
        )
        self.preference_pairs.append(preference)
        
        # Auto-save if storage path is set
        if self.storage_path:
            self.save_to_file(self.storage_path)
        
        return preference
    
    def get_feedback_count(self) -> int:
        """Get total number of feedback entries."""
        return len(self.feedback_history)
    
    def get_preference_count(self) -> int:
        """Get total number of preference pairs."""
        return len(self.preference_pairs)
    
    def get_corrections(self) -> List[HumanFeedback]:
        """Get all feedback entries where model prediction != human label."""
        return [f for f in self.feedback_history 
                if f.model_prediction != f.human_label]
    
    def export_to_dataset(self) -> List[Tuple[str, int]]:
        """
        Export human feedback to dataset format.
        
        Returns:
            List of (url_or_domain, label) tuples based on human labels
        """
        return [(f.url_or_domain, f.human_label) for f in self.feedback_history]
    
    def save_to_file(self, filepath: str):
        """
        Save feedback to JSON file.
        
        Args:
            filepath: Path to JSON file
        """
        data = {
            'feedback_history': [f.to_dict() for f in self.feedback_history],
            'preference_pairs': [
                {
                    'item_a': p.item_a,
                    'item_b': p.item_b,
                    'preferred': p.preferred,
                    'context': p.context,
                    'timestamp': p.timestamp.isoformat() if p.timestamp else None,
                    'user_id': p.user_id
                }
                for p in self.preference_pairs
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """
        Load feedback from JSON file.
        
        Args:
            filepath: Path to JSON file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.feedback_history = [
                HumanFeedback.from_dict(f_dict)
                for f_dict in data.get('feedback_history', [])
            ]
            
            self.preference_pairs = []
            for p_dict in data.get('preference_pairs', []):
                if 'timestamp' in p_dict and p_dict['timestamp']:
                    p_dict['timestamp'] = datetime.fromisoformat(p_dict['timestamp'])
                self.preference_pairs.append(PreferencePair(**p_dict))
        except FileNotFoundError:
            # File doesn't exist yet, start with empty collections
            pass


class RLHFInterface(ABC):
    """
    Abstract base class for RLHF/RLHP integration.
    
    This interface defines the contract for implementing Reinforcement Learning
    from Human Feedback/Preferences. Subclasses should implement methods for
    training reward models, optimizing policies, and integrating feedback.
    
    This is a placeholder for future RLHF/RLHP implementation. Extend this class
    to implement specific RLHF algorithms (e.g., PPO, DPO).
    """
    
    @abstractmethod
    def train_reward_model(self, feedback_collector: FeedbackCollector) -> tf.keras.Model:
        """
        Train a reward model from human feedback.
        
        Args:
            feedback_collector: FeedbackCollector with human feedback data
            
        Returns:
            Trained reward model (TensorFlow/Keras model)
        """
        pass
    
    @abstractmethod
    def optimize_policy(
        self,
        base_model: tf.keras.Model,
        reward_model: tf.keras.Model,
        feedback_collector: FeedbackCollector
    ) -> tf.keras.Model:
        """
        Optimize policy using reinforcement learning.
        
        Args:
            base_model: Base classifier model to optimize
            reward_model: Reward model trained on human feedback
            feedback_collector: FeedbackCollector with feedback data
            
        Returns:
            Optimized model
        """
        pass
    
    @abstractmethod
    def compute_reward(
        self,
        reward_model: tf.keras.Model,
        predictions: np.ndarray,
        features: np.ndarray,
        feedback: Optional[HumanFeedback] = None
    ) -> np.ndarray:
        """
        Compute rewards using reward model.
        
        Args:
            reward_model: Trained reward model
            predictions: Model predictions
            features: Input features
            feedback: Optional human feedback for comparison
            
        Returns:
            Reward scores
        """
        pass


# Placeholder for future RLHF/RLHP implementations
# Example structure (not functional):
#
# class PPO_RLHF(RLHFInterface):
#     """Proximal Policy Optimization implementation of RLHF."""
#     def train_reward_model(self, feedback_collector):
#         # Implementation for training reward model
#         pass
#
#     def optimize_policy(self, base_model, reward_model, feedback_collector):
#         # PPO implementation
#         pass
#
#     def compute_reward(self, reward_model, predictions, features, feedback):
#         # Reward computation
#         pass


# ============================================================================
# Main Execution Script
# ============================================================================

if __name__ == '__main__':
    """
    Main execution script to download ad-blocker lists, generate dataset,
    train the model, and save it.
    
    Usage:
        python adblocker_ai.py
    """
    import os
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train TensorFlow ad-blocker model from Adblock Plus lists'
    )
    parser.add_argument(
        '--list-urls',
        type=str,
        nargs='+',
        default=None,
        help='URLs of Adblock Plus format lists to download (default: uses parser defaults)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples per class (default: use all available)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='./adblocker_model',
        help='Directory to save the trained model (default: ./adblocker_model)'
    )
    parser.add_argument(
        '--use-domain-features',
        action='store_true',
        help='Use domain-only features (8 dim) instead of URL features (25 dim)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TensorFlow Ad-Blocker AI Model Training")
    print("=" * 70)
    print()
    
    # Step 1: Download and parse ad-blocker lists
    print("Step 1: Downloading and parsing ad-blocker lists...")
    print("-" * 70)
    list_parser = AdBlockListParser()
    
    if args.list_urls:
        parsed_data = list_parser.download_and_parse(args.list_urls)
    else:
        parsed_data = list_parser.download_and_parse()
    
    print(f"\nTotal parsed data:")
    print(f"  Domains: {len(parsed_data['domains'])}")
    print(f"  URL patterns: {len(parsed_data['url_patterns'])}")
    print(f"  Regex patterns: {len(parsed_data['regex_patterns'])}")
    print(f"  Domain patterns: {len(parsed_data['domain_patterns'])}")
    print()
    
    # Step 2: Generate dataset
    print("Step 2: Generating labeled dataset...")
    print("-" * 70)
    dataset = generate_dataset(
        parser=list_parser,
        max_samples_per_class=args.max_samples,
        balance_dataset=True,
        seed=args.seed
    )
    
    if not dataset:
        print("Error: Failed to generate dataset. Please check list downloads.")
        exit(1)
    
    print(f"Generated dataset with {len(dataset)} samples")
    label_counts = {}
    for _, label in dataset:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"  Class distribution: {label_counts}")
    print()
    
    # Step 3: Prepare training data
    print("Step 3: Preparing training data (feature extraction, normalization, splitting)...")
    print("-" * 70)
    use_url_features = not args.use_domain_features
    input_dim = 25 if use_url_features else 8
    
    X_train, y_train, X_val, y_val, X_test, y_test, norm_stats = prepare_training_data(
        dataset=dataset,
        feature_extractor=None,  # Create new FeatureExtractor
        use_url_features=use_url_features,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize=True,
        stratify=True,
        seed=args.seed,
        verbose=1
    )
    print()
    
    # Step 4: Build and train model
    print("Step 4: Building and training model...")
    print("-" * 70)
    model = AdBlockerModel(input_dim=input_dim)
    
    print("\nModel architecture:")
    print(model.get_model_summary())
    print()
    
    print(f"Training model for {args.epochs} epochs with batch size {args.batch_size}...")
    history = model.train(
        X=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    print()
    
    # Step 5: Evaluate on test set
    print("Step 5: Evaluating on test set...")
    print("-" * 70)
    test_results = model.evaluate(X_test, y_test, verbose=1)
    print("\nTest set metrics:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Step 6: Save model
    print("Step 6: Saving model...")
    print("-" * 70)
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_model(args.model_dir)
    print(f"Model saved to: {args.model_dir}")
    print()
    
    # Step 7: Test predictions
    print("Step 7: Testing predictions on sample URLs...")
    print("-" * 70)
    test_urls = [
        "https://example.com/ads/banner.jpg",
        "https://doubleclick.net/tracker.js",
        "https://google.com/search",
        "https://github.com/user/repo",
        "https://ads.example.com/track",
    ]
    
    for url in test_urls:
        pred, conf = model.predict(url)
        label = "AD" if pred == 1 else "LEGITIMATE"
        print(f"  {url}")
        print(f"    Prediction: {label} (confidence: {conf:.4f})")
    print()
    
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nTo use the model in your code:")
    print(f"  from adblocker_ai import AdBlockerModel")
    print(f"  model = AdBlockerModel(input_dim={input_dim})")
    print(f"  model.load_model('{args.model_dir}')")
    print(f"  prediction, confidence = model.predict('https://example.com')")
    print()
