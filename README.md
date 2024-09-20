# ist_dash_2024_rec

Repo for group project in Data Science Engineering.

Desired Characterics of Datasets:

- Unbalanced dataset (more than 80% of data is class 0 (negative class))
- High number of variables (more than 50 variables)
- Simbolic vs numerical variables.

This a classification task project.
Task description: https://e.tecnicomais.pt/pluginfile.php/340884/mod_resource/content/1/20240909.enunciadoProjDS.pdf

It will use 3 out of the following datasets:

1. Online Behaviour data from multiple category store ✅

   - https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
   - Collaborative + Content Based Recommender System for ecommerce dataset
   - based on ecommerce events (product page view, add to cart, purchase) + catalogue of the store.
   - 285M events although partitioned by year/months periods
   - class label target should be purchase or add to cart. Research dataset.
   - Simbolic vs numeric variables
   - Hugging Face Dataset: https://huggingface.co/datasets/jin-ying-so-cute/ecommerce-user-behavior-data 34M records

2. Google Analytics Sample Dataset Merchandise Store ✅
  - https://www.kaggle.com/datasets/bigquery/google-analytics-sample
  - dataset link: https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset
  
  - interesting use case for multiple variables (device, time, page, channel, etc)
  - interesting example on returning https://www.kaggle.com/code/chewannsiong/tensorflow-predict-website-revisit


3. Criteo Advertising Click and/or Conversion Prediction ✅
   - https://huggingface.co/datasets/reczoo/Criteo_x1
   - The Criteo dataset is a widely-used benchmark dataset for CTR prediction, which contains about one week of click-through data for display advertising. It has 13 numerical feature fields and 26 categorical feature fields.



# Brainstorming Cases (can be used in the future)

- Airbnb Deep Learning Case? Relation between image, reviews, host, price, location to find a good deal

- Real State Use Cases

  - Real State Binary Price Tier Prediction:
    - Class Target: Predict house is above or below a price tier. Example: below or above 200k€
    - or multiple price tiers if possible.
    - Dataset: Idealista, Casafari, imovirtual (to investigate, scrape, kaggle)

- Used Car Use Cases

  - Used Car Binary Price Tier Prediction:
    - Class Target: Predict house is above or below a price tier. Example: below or above 50k€
    - or multiple price tiers if possible.
    - Dataset: AutoScout24, StandVirtual, Mercedes Benz Used Vans

- Google Analytics Sample Dataset Merchandise Store (investigate dataset export and values)
  - https://www.kaggle.com/datasets/bigquery/google-analytics-sample
  - interesting use case for multiple variables (device, time, page, channel, etc)
  - interesting example on returning https://www.kaggle.com/code/chewannsiong/tensorflow-predict-website-revisit



- Amazon|Aliexpress|Shein|Temu ratings (✅ but research dataset or scrape)

   - Dataset with ratings 1-5 stars, transform in binary for classification task.
   - class label target should be GOOD or BAD product review.
     - assume 4-5 stars is good and 1-2 stars is bad
     - analyze if we can do with 3 classes: bad, neutral and good.
     - can detect fraudulent or good products.
   - predictive model to classify rating of product (without or low number of ratings) as positive or negative.
     - TO VERIFY: if the product does not have ratings what is the chance that it is good or bad? should we use number of ratings as variable as well as reviews themselves?
   - can have a high number of variables depending on product category attributes
   - Dataset Example: https://www.kaggle.com/datasets/sayedmahmoud/amazanreviewscor5
   - Dataset https://nijianmo.github.io/amazon/index.html
   - Dataset https://amazon-reviews-2023.github.io/