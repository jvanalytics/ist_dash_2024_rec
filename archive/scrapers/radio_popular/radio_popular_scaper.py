from bs4 import BeautifulSoup as bs
import re
import requests
from datetime import datetime
import csv



headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}


sitemaps = ['https://lojae-s3-prd-files.radiopopular.pt/files/sitemaps/sitemap-produtos-00.xml',
            'https://lojae-s3-prd-files.radiopopular.pt/files/sitemaps/sitemap-produtos-01.xml']


# test=False to place in Prod

# test=True
test=False

if test==False:


    crawl_list = []

    for i in sitemaps:
        try:
            response = requests.get(
                i, headers=headers)
            soup = bs(response.content, 'xml')
            urls = soup.find_all('url')

            for i in urls:
                product = i.find('loc').text
                crawl_list.append(product)
        except requests.exceptions.ConnectionError as err:
            print(str(i)+" error")

    print(str(len(crawl_list)) + " links in sitemap")

else:
    crawl_list = ['https://www.radiopopular.pt/produto/pc-portatil-asus-fa507nv-r57a46cb1','https://www.radiopopular.pt/produto/rato-asus-rog-strix-carry?bundleReferral=p-bdl-115202-bdl']




def crawl(link):
    try:
        response = requests.get(
            link, headers=headers)
        soup = bs(response.text, 'lxml')

        try:
            title = soup.find('h1').text.strip()
        except:
            title = None

        try:
            flix = str(soup.find('section', 'flix'))
            ean = re.search(r'data.flix.ean..\d+',
                            flix).group().replace('data-flix-ean="', '')
            brand = re.search(r'data.flix.brand..\w+(\s|.\w+)',
                              flix).group().replace('data-flix-brand="', '')
        except:
            ean = None
            brand = None
        try:
            sku = soup.find('div', 'sa selectable fl').text
        except:
            sku = None

        try:
            main_price = soup.find('div', 'price notranslate').text.replace(
                '.', '').replace(',', '.')
        except:
            main_price = None

        try:
            pvp = soup.find('span', 'no-strike').text.replace('PVPR* €',
                                                              '').replace(
                '.', '').replace(',', '.')
        except:
            pvp = ''

        try:
            features = soup.find('div', 'features').text
        except:
            features = None

        try:
            buy = soup.find(
                'div', 'button buy fl uppercase center-text desktop-only visible').text
            if buy == 'comprar':
                stock = 'available'
        except:
            stock = 'out_of_stock'
        try:
            image_url = soup.find('div', 'images wrapper fl cb').find(
                'div', 'module thumbnail fl').get('style').replace('background-image:url(', '').replace(')', '').replace(';', '').replace("'", "")
        except:
            image_url = None

        info = {'date': str(datetime.now()), 'url': str(link), 'sku': sku, 'ean': ean, 'title': title, 'main_price': main_price,
                'pvp': pvp, 'brand': brand, 'stock': stock, 'features': features, 'image_url': image_url}

    except requests.exceptions.ConnectionError as err:
        print(str(link) + " error: ")
    return info


# Generate the filename with the current date
filename = f"radio_popular_data_{datetime.now().strftime('%Y-%m-%d')}.csv"

# Get the fieldnames from the keys of the info dictionary
fieldnames = [
    'date',
    'url',
    'sku',
    'ean',
    'title',
    'main_price',
    'pvp',
    'brand',
    'stock',
    'features',
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