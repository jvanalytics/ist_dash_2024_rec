from bs4 import BeautifulSoup as bs
import re
import requests
from datetime import datetime
import csv



# Define headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}


# test=False to place in Prod
# test=True
test=False

if test==False:

    # Define sitemaps
    sitemaps = ['https://www.continente.pt/sitemap-custom_sitemap_1-product.xml', 'https://www.continente.pt/sitemap-custom_sitemap_4-product.xml',
                'https://www.continente.pt/sitemap-custom_sitemap_8-product.xml']

    # List to store product URLs
    crawl_list = []

    # Loop through sitemaps to gather product URLs
    for i in sitemaps:
        try:
            response = requests.get(i, headers=headers)
            soup = bs(response.content, 'xml')
            urls = soup.find_all('url')

            for url in urls:
                product = url.find('loc').text
                crawl_list.append(product)

        except requests.exceptions.ConnectionError as err:
            print(f"{i} error: {err}")

    print(f"{len(crawl_list)} links in sitemap")

else:
    crawl_list=['https://www.continente.pt/produto/bebida-vegetal-de-aveia-barista-rude-health-7777742.html','https://www.continente.pt/produto/iogurte-liquido-magro-morango-continente-equilibrio-5890060.html','https://www.continente.pt/produto/peito-de-frango-em-vacuo-continente-6950358.html','https://www.continente.pt/produto/frango-inteiro-continente-4234958.html','https://www.continente.pt/produto/porcao-de-salmao-fresco-6661780.html']



# Function to scrape individual product pages
def crawl(link):
    try:
        response = requests.get(link, headers=headers)
        soup = bs(response.text, 'lxml')
        
        # Extract SKU from URL
        try:
            sku = re.search(r'\d+\.html$', link).group().replace('.html', '')
        except AttributeError:
            sku = None
        
        # Extract product information
        try:
            title = soup.find('h1').text.strip()
        except AttributeError:
            title = None

        if title:
            try:
                main_price = soup.find('span', 'ct-price-formatted').text.replace('€', '').replace(',', '.').strip()
            except AttributeError:
                main_price = None

            try:
                unit_price = soup.find('span', 'ct-price-value').text.replace('€', '').replace(',', '.').strip()
            except AttributeError:
                unit_price = None

            try:
                price_unit = soup.find('span', 'pwc-m-unit').text.replace('/', '')
            except AttributeError:
                price_unit = None


            try:
                pvp = soup.find('p', 'pwc-discount-amount pwc-discount-amount-pvpr col-discount-amount').text.replace(
                    'PVP Recomendado: €', '').replace('/un', '').replace(',', '.')
            except AttributeError:
                pvp = None

            try:
                ean = soup.find('div', 'row no-gutters js-product-tabs--wrapper').find(
                    'a', 'js-details-header js-nutritional-tab-anchor d-none').get('data-url')
                ean_fix = re.search(r'ean=\d+', ean).group().replace('ean=', '')
            except AttributeError:
                ean_fix = None

            try:
                brand = soup.find('a', 'ct-pdp--brand').text.strip()
            except AttributeError:
                brand = None

            try:
                short_description = soup.find('div', 'ct-pdp--short-description').text.strip()
            except AttributeError:
                short_description = None

            # try:
            #     long_description = soup.find('div', 'ct-pdp--description-content col-pdp--description-content').text.strip()
            #     long_description = long_description.replace('\n', ' ').replace('\r', '').strip()

            # except AttributeError:
            #     long_description = None

            try:
                breadcrumb_url = soup.find('ul', 'breadcrumbs').find(
                    'a', 'pwc-anchor col-anchor breadcrumb-anchor').get('href')
            except AttributeError:
                breadcrumb_url = None

            try:
                image_url = soup.find('div', 'pdp-img-container').find('img', 'ct-product-image').get('src')
            except AttributeError:
                image_url = None


            # Collect the product info into a dictionary
            info = {
                'date': str(datetime.now()), 
                'url': link, 'sku': sku, 
                'ean': ean_fix, 
                'title': title,
                'main_price': main_price, 
                'unit_price': unit_price, 
                'price_unit': price_unit,
                'pvp': pvp, 'brand': brand, 
                'short_description': short_description,
                # 'long_description':long_description,
                'breadcrumb_url': breadcrumb_url, 
                'image_url': image_url
            }
        else:
            # If title is None, set all fields to None
            info = {
                'date': str(datetime.now()), 
                'url': link, 
                'sku': sku, 
                'ean': None, 
                'title': None, 
                'main_price': None,
                'unit_price': None, 
                'price_unit':None,
                'pvp': None, 
                'brand': None,
                'short_description': None,
                # 'long_description':None,
                'breadcrumb_url': None,
                'image_url': None
            }

    except requests.exceptions.ConnectionError as err:
        print(f"{link} error: {err}")
        return None
    
    return info



# Generate the filename with the current date
filename = f"continente_data_{datetime.now().strftime('%Y-%m-%d')}.csv"

# Get the fieldnames from the keys of the info dictionary
fieldnames = [
    'date',
    'url',
    'sku',
    'ean',
    'title',
    'main_price',
    'unit_price',
    'price_unit',
    'pvp',
    'brand',
    'short_description',
    # 'long_description',
    'breadcrumb_url',
    'image_url'
]

# Open the CSV file in write mode or append mode if you want to keep adding new data
if test==False:
    csv_mode='a'
else:
    csv_mode='w'


with open(filename, mode=csv_mode, newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # Write the header
    writer.writeheader()

    # Loop through the product URLs and scrape the data
    for url in crawl_list:
        product_info = crawl(url)
        if product_info:
            writer.writerow(product_info)
            print(f'Data written for URL: {url}')

print(f"Data saved to {filename}")