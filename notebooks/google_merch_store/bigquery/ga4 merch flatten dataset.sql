DECLARE start_date DATE DEFAULT '2020-01-01';
DECLARE end_date DATE DEFAULT '2021-12-31';

CREATE OR REPLACE TABLE `ist-data-science.ga4_google_merch_store.flattened_raw` 
 (
  event_date DATE
  , session_id INT64
  , user_pseudo_id STRING
  , event_name STRING
  , event_timestamp INT64
  , page_location  STRING
  , page_title   STRING
  , device_category  STRING
  , device_mobile_brand_name  STRING
  , device_mobile_model_name  STRING
  , device_mobile_marketing_name  STRING
  , device_operating_system  STRING
  , device_operating_system_version STRING
  , device_language  STRING
  , device_is_limited_ad_tracking STRING
  , device_web_info_browser STRING
  , device_web_info_browser_version STRING
  , geo_continent STRING
  , geo_country STRING
  , geo_region STRING
  , geo_city STRING
  , geo_sub_continent STRING
  , geo_metro STRING
  , traffic_source_name STRING
  , traffic_source_medium STRING
  , traffic_source_source STRING
  , page_referrer STRING
  , entrances INT64
  , debug_mode INT64
  , ga_session_number INT64
  , session_engaged INT64
  , engagement_time_msec INT64
  , ecommerce_total_item_quantity INT64
  , ecommerce_purchase_revenue FLOAT64
  , ecommerce_shipping_value FLOAT64
  , ecommerce_tax_value FLOAT64
  , ecommerce_unique_items INT64
  , ecommerce_transaction_id STRING
  , item_id STRING
  , item_name  STRING
  , item_brand  STRING
  , item_variant  STRING
  , item_category  STRING
  , price FLOAT64
  , quantity INT64
  , item_revenue FLOAT64
  , item_list_index INT64
  , promotion_name   STRING
)
PARTITION BY event_date


AS (
WITH all_events AS (

SELECT 
   CAST(FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', event_date)) AS DATE) AS event_date
  , (select value.int_value from unnest(event_params) where key = 'ga_session_id') as session_id
  , user_pseudo_id
  , event_name
  , event_timestamp
  , (select value.string_value from unnest(event_params) where key = 'page_location') as page_location
  , (select value.string_value from unnest(event_params) where key = 'page_title') as page_title  

  , device.category as device_category
  , device.mobile_brand_name as device_mobile_brand_name
  , device.mobile_model_name as device_mobile_model_name
  , device.mobile_marketing_name as device_mobile_marketing_name
  , device.operating_system as device_operating_system
  , device.operating_system_version as device_operating_system_version
  , device.language as device_language
  , device.is_limited_ad_tracking as device_is_limited_ad_tracking
  , device.web_info.browser as device_web_info_browser
  , device.web_info.browser_version as device_web_info_browser_version
  , geo.continent as geo_continent
  , geo.country as geo_country
  , geo.region as geo_region
  , geo.city as geo_city
  , geo.sub_continent as geo_sub_continent
  , geo.metro as geo_metro
  , traffic_source.name as traffic_source_name
  , traffic_source.medium as traffic_source_medium
  , traffic_source.source as traffic_source_source
  ,(select value.string_value from unnest(event_params) where key = 'page_referrer') as page_referrer
  ,(select value.int_value from unnest(event_params) where key = 'entrances') as entrances
  ,(select value.int_value from unnest(event_params) where key = 'debug_mode') as debug_mode
  ,(select value.int_value from unnest(event_params) where key = 'ga_session_number') as ga_session_number
  ,CAST((select value.string_value from unnest(event_params) where key = 'session_engaged') AS INT64) as session_engaged
  ,(select value.int_value from unnest(event_params) where key = 'engagement_time_msec') as engagement_time_msec
  , ecommerce.total_item_quantity as ecommerce_total_item_quantity
  , ecommerce.purchase_revenue as ecommerce_purchase_revenue
  , ecommerce.shipping_value as ecommerce_shipping_value
  , ecommerce.tax_value as ecommerce_tax_value
  , ecommerce.unique_items as ecommerce_unique_items
  , ecommerce.transaction_id as ecommerce_transaction_id   

FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` 

WHERE _TABLE_SUFFIX BETWEEN format_date('%Y%m%d',start_date) AND format_date('%Y%m%d',end_date)

),


ecommerce_events AS (

SELECT 
  (select value.int_value from unnest(event_params) where key = 'ga_session_id') as session_id
  , event_name
  , event_timestamp
  , items.item_id
  , items.item_name
  , items.item_brand
  , items.item_variant
  , items.item_category
  , items.price
  , items.quantity
  , IF(items.item_revenue IS NULL,0,items.item_revenue) as item_revenue
  , CAST(
    (CASE 
      WHEN items.item_list_index LIKE 'Slide%' THEN REGEXP_EXTRACT(items.item_list_index, r"\d")
      WHEN items.item_list_index LIKE '(not set)' THEN NULL
      ELSE items.item_list_index
    END)
      AS INT64) as item_list_index
  , items.promotion_name  

FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` ,
UNNEST (items) as items

WHERE _TABLE_SUFFIX BETWEEN format_date('%Y%m%d',start_date) AND format_date('%Y%m%d',end_date)

)


SELECT
  ae.event_date 
  , ae.session_id 
  , user_pseudo_id 
  , ae.event_name 
  , ae.event_timestamp 
  , page_location  
  , page_title   
  , device_category  
  , device_mobile_brand_name  
  , device_mobile_model_name  
  , device_mobile_marketing_name  
  , device_operating_system  
  , device_operating_system_version 
  , device_language  
  , device_is_limited_ad_tracking 
  , device_web_info_browser 
  , device_web_info_browser_version 
  , geo_continent 
  , geo_country 
  , geo_region 
  , geo_city 
  , geo_sub_continent 
  , geo_metro 
  , traffic_source_name 
  , traffic_source_medium 
  , traffic_source_source 
  , page_referrer 
  , entrances 
  , debug_mode 
  , ga_session_number 
  , session_engaged 
  , engagement_time_msec 
  , ecommerce_total_item_quantity 
  , ecommerce_purchase_revenue 
  , ecommerce_shipping_value 
  , ecommerce_tax_value 
  , ecommerce_unique_items 
  , ecommerce_transaction_id 
  , item_id 
  , item_name  
  , item_brand  
  , item_variant  
  , item_category  
  , price 
  , quantity 
  , item_revenue 
  , item_list_index 
  , promotion_name   
FROM 
all_events as ae
LEFT JOIN ecommerce_events as e
  ON ae.event_name = e.event_name
  AND ae.event_timestamp = e.event_timestamp
  AND ae.session_id = e.session_id
)