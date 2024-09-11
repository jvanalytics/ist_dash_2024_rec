# ist_dash_2024_rec
repo for group project in Data Science Engineering.

Desired Characterics of Datasets:
- Unbalanced dataset (more than 80% of data is class 0 (negative class))
- High number of variables (more than 50 variables)
- Simbolic vs numerical variables. 

We intend to deliver a recommendation engine system for ecommerce and automotive companies.
The project uses three datasets:



1. Online Behaviour data from multiple category store 
    - https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store 
    - Collaborative + Content Based Recommender System for ecommerce dataset
    - based on ecommerce events (product page view, add to cart, purchase) + catalogue of the store.
    - 285M events although partitioned by year/months periods
    - class target should be purchase or add to cart. Research dataset.


2. Amazon|Aliexpress|Shein|Temu ratings (to research dataset or scrape)
    - Dataset with ratings 1-5 stars, transform in binary for classification task. 
    - example: product ratings of products as positive or negative.
    - predictive model to classify rating of product (without or low number of ratings) as positive or negative.
    - can detect fraudulent or good products.
    - analyze if we can do with 3 classes: bad, neutral and good.

3. Research dataset with car industry