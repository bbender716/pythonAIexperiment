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
            'https://big.oisd.nl',
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
