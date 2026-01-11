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
