import scrapy


class ResultsPageSpider(scrapy.Spider):
    name = "results_page"
    allowed_domains = ["auto24.com"]
    start_urls = [f"https://www.autoscout24.com/lst?atype=C&desc=0&sort=standard&page={page}" for page in range(1, 22)]

    def parse(self, response):
        article_list=response.css('article')


        for article in article_list:
                    # Extract the text content of each <h2> element
                    title = article.css('h2::text').get()
                    subtitle = article.css('span.ListItem_version__5EWfi::text').get()
                    article_id = article.attrib['id']
                    price_label=article.css('::attr(data-price-label)').get()
                    href = 'https://www.autoscout24.com'+article.css('a.ListItem_title__ndA4s::attr(href)').get()
                    model = article.css('article::attr(data-model)').get()
                    date = article.css('span[data-testid="VehicleDetails-calendar"]::text').get()
                    kms = article.css('span[data-testid="VehicleDetails-mileage_road"]::text').get()
                    price = article.css('p[data-testid="regular-price"]::text').get().replace('€', '').replace(',', '').replace('.', '').replace('-', '').strip()
                    image_url = article.css('picture source::attr(srcset)').get()
                    power = article.css('span[data-testid="VehicleDetails-speedometer"]::text').get()
                    fuel = article.css('span[data-testid="VehicleDetails-gas_pump"]::text').get()
                    transmission = article.css('span[data-testid="VehicleDetails-transmission"]::text').get()
                    seller_name = article.css('span[data-testid="sellerinfo-company-name"]::text').get()
                    seller_address = article.css('span[data-testid="sellerinfo-address"]::text').get()
                    

                    yield {
                        'title': title,
                        'subtitle': subtitle,
                        'article_id': article_id,
                        'price_label': price_label,
                        'href': href,
                        'model': model,
                        'href': href,
                        'model': model,
                        'date': date,
                        'kms': kms,
                        'price': price,
                        'image_url': image_url,
                        'power': power,
                        'fuel': fuel,
                        'transmission': transmission,
                        'seller_name': seller_name,
                        'seller_address': seller_address,

                        # Add more fields as needed
                    }


        # Extract the current page number from the URL
        current_page = int(response.url.split('page=')[-1])

        # If the current page number is less than 20, generate the next page URL
        if current_page < 22:
            next_page_number = current_page + 1
            next_page_url = f"https://www.autoscout24.com/lst?atype=C&desc=0&sort=standard&page={next_page_number}"
            yield scrapy.Request(url=next_page_url, callback=self.parse)
