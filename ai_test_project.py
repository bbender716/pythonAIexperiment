import requests
from bs4 import BeautifulSoup
import time
import re
from typing import Optional, Dict

class AmazonPriceScraper:
    """
    Amazon price scraper using requests and BeautifulSoup.
    Note: Amazon has anti-scraping measures, so this may not work for all pages.
    For more reliable scraping, use Selenium (see selenium_scraper.py below).
    """
    
    def __init__(self):
        # Headers to mimic a real browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def get_price(self, url: str) -> Optional[Dict[str, str]]:
        """
        Scrape price information from an Amazon product page.
        
        Args:
            url: Amazon product URL
            
        Returns:
            Dictionary with product title, price, and availability, or None if failed
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple selectors for price (Amazon changes these frequently)
            price = None
            
            # Common price selectors
            price_selectors = [
                'span.a-price-whole',
                'span#priceblock_ourprice',
                'span#priceblock_dealprice',
                'span.a-price .a-offscreen',
                '.a-price[data-a-size="xl"] .a-offscreen',
                '#price',
            ]
            
            for selector in price_selectors:
                price_element = soup.select_one(selector)
                if price_element:
                    price_text = price_element.get_text(strip=True)
                    # Extract numeric price
                    price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
                    if price_match:
                        price = f"${price_match.group()}"
                        break
            
            # Get product title
            title = None
            title_selectors = [
                '#productTitle',
                'h1.a-size-large',
                'span#productTitle',
            ]
            
            for selector in title_selectors:
                title_element = soup.select_one(selector)
                if title_element:
                    title = title_element.get_text(strip=True)
                    break
            
            # Check availability
            availability = None
            availability_selectors = [
                '#availability span',
                '#availability',
                '.a-color-success',
            ]
            
            for selector in availability_selectors:
                avail_element = soup.select_one(selector)
                if avail_element:
                    availability = avail_element.get_text(strip=True)
                    break
            
            if price or title:
                return {
                    'title': title or 'N/A',
                    'price': price or 'N/A',
                    'availability': availability or 'N/A',
                    'url': url
                }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
        except Exception as e:
            print(f"Error parsing page: {e}")
        
        return None


def scrape_amazon_price_simple(url: str) -> Optional[Dict[str, str]]:
    """
    Simple function to scrape a single Amazon product price.
    
    Example usage:
        result = scrape_amazon_price_simple("https://www.amazon.com/dp/B08N5WRWNW")
        print(result)
    """
    scraper = AmazonPriceScraper()
    return scraper.get_price(url)


# Example usage
if __name__ == '__main__':
    # Example: Replace with actual Amazon product URL
    product_url = "https://www.amazon.com/s?k=questbar&crid=37FPP8KM0OJUD&sprefix=questba%2Caps%2C151" # Example product URL
    
    print("Scraping Amazon price...")
    result = scrape_amazon_price_simple(product_url)
    
    if result:
        print(f"\nProduct: {result['title']}")
        print(f"Price: {result['price']}")
        print(f"Availability: {result['availability']}")
    else:
        print("Failed to scrape price. Amazon may be blocking requests or the page structure has changed.")
        print("\nTip: Consider using Selenium for more reliable scraping (see alternative method below).")
    
    # Be respectful: add delay between requests
    time.sleep(2)