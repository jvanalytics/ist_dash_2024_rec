# CONDA ENVIRONMENT activation and libraries:

1. clone your environment

   - conda create --name ist --clone base

2. to export your environment

   - conda env export > ist_environment.yml

3. to install other enironment:
   - conda env create -f ist_environment.yml

This will ensure that you can run the same libraries in your machine.
You will need to select it as Python Kernel when running python or jupyter notebooks

# scrapy notes:

    - input in terminal on project folder "folder_name tree" to see folder structure from scrapy
    - input in terminal on project folder write "folder_name scrapy shell" to go to scrapy shell that allows to debug the spider in a more interactive way than just using print statements.
        Example: response.css('h2::text').get() will return the first h2 tag's of the response
    - to debug output of scrapy crawl input "scrapy crawl name_of_spider"
    - to get an output with file of scrapy crawl input "scrapy crawl name_of_spider -o filename.json"

# Hosting and Reading Datasets Online

1. Hugging Face hosted

Pandas Dataframe from link in dataset
Example: https://huggingface.co/datasets/yashraizad/yelp-open-dataset-reviews?row=0
read from Pandas
df = pd.read_parquet("hf://datasets/yashraizad/yelp-open-dataset-reviews/review.parquet")

# rapids cudf activation

1. to activate conda rapids inser command in unbuntu
   (base)PATH:~$ conda activate rapids-24.08

2. then to activate jupyter lab server:
   (rapids-24.08) PATH:~$ jupyter lab --no-browser --ip=0.0.0.0 --port=8888

3. copy url token after into VSC:
   http://127.0.0.1:8888/lab?token=4ca863e01b0c955444c2b00ab444e6f0c4798a7c9c778d9c

4. after using input command in powershell
   wsl --terminate Ubuntu
   OR
   wsl --shutdown
