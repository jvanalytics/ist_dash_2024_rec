# ist_dash_2024_rec
repo for group project in Data Science Engineering.

Desired Characterics of Datasets:
- Unbalanced dataset (more than 80% of data is class 0 (negative class))
- High number of variables (more than 50 variables)
- Simbolic vs numerical variables. 

We intend to deliver a recommendation engine system for ecommerce and automotive companies.
The project uses three datasets:



1. Online Behaviour data from multiple category store ✅
    - https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store 
    - Collaborative + Content Based Recommender System for ecommerce dataset
    - based on ecommerce events (product page view, add to cart, purchase) + catalogue of the store.
    - 285M events although partitioned by year/months periods
    - class target should be purchase or add to cart. Research dataset.


2. Amazon|Aliexpress|Shein|Temu ratings (✅ but research dataset or scrape) 
    - Dataset with ratings 1-5 stars, transform in binary for classification task. 
    - example: product ratings of products as positive or negative.
    - assume a 4-5star product
    - predictive model to classify rating of product (without or low number of ratings) as positive or negative.
    - class target should be GOOD or BAD product.
        - analyze if we can do with 3 classes: bad, neutral and good.
        - can detect fraudulent or good products.
    - Dataset Example: https://www.kaggle.com/datasets/sayedmahmoud/amazanreviewscor5

3. Google Analytics Sample Dataset Merchandise Store (investigate dataset export and values)
    - https://www.kaggle.com/datasets/bigquery/google-analytics-sample
    - interesting use case for multiple variables (device, time, page, channel, etc)
    - interesting example on returning https://www.kaggle.com/code/chewannsiong/tensorflow-predict-website-revisit

4. Real State Use Cases
    - Real State Binary Price Tier Prediction: 
        - Class Target: Predict house is above or below a price tier. Example: below or above 200k€
        - or multiple price tiers if possible. 
        - Dataset: Idealista, Casafari, imovirtual (to investigate, scrape, kaggle)

5. Used Car Use Cases
    - Used Car Binary Price Tier Prediction: 
        - Class Target: Predict house is above or below a price tier. Example: below or above 50k€
        - or multiple price tiers if possible. 
        - Dataset: AutoScout24, StandVirtual, Mercedes Benz Used Vans



Brainstorming Cases (can be used in the future)
    - Airbnb Deep Learning Case? Relation between image, reviews, host, price, location to find a good deal
