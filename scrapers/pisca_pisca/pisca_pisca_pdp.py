from bs4 import BeautifulSoup as bs
import re
import requests
from datetime import datetime

# Define headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}

url='https://www.piscapisca.pt/carros/usados/ford-kuga-diesel/0zAr1'

# ---------------- NOT WORKING ---- NEED TO DO JAVASCRIPT RENDERING WITH SELENIUM OR SUCH

def crawl(link):
    try:
        response = requests.get(link, headers=headers)
        soup = bs(response.text, 'html')

        # Extract SKU from URL which is the last part after slash
        try:
            sku = re.search(r'/([^/]+)$', link).group(1)
        except AttributeError:
            sku = None
        
        # Extract product information
        try:
            title = soup.find('h1', class_='title')
        except AttributeError:
            title = None

        if title:
            try:
                main_price = soup.find('span', 'price').text.replace('€', '').replace(' ', '').strip()
            except AttributeError:
                main_price = None

            try:
                year = soup.find('span', '_ngcontent-ssr-app-c153').text.strip()
            except AttributeError:
                year = None


            # Collect the product info into a dictionary
            info = {
                'date': str(datetime.now()), 
                'url': link, 
                'sku': sku, 
                'title': title,
                'main_price': main_price, 
                'year': year, 
            }
        else:
            # If title is None, set all fields to None
            info = {
                'date': str(datetime.now()), 
                'url': link, 
                'sku': sku, 
                'title': None,
                'main_price': None, 
                'year': None, 

            }

    except requests.exceptions.ConnectionError as err:
        print(f"{link} error: {err}")
        return None
    
    return info


link_scrape=crawl(url)

print(link_scrape)