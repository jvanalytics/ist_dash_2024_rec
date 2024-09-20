DECLARE start_date DATE DEFAULT '2020-01-01';
DECLARE end_date DATE DEFAULT '2021-12-31';



SELECT 
   FORMAT_DATE('%Y-%m-%d', PARSE_DATE('%Y%m%d', event_date)) AS event_date
  , (select value.int_value from unnest(event_params) where key = 'ga_session_id') as session_id
  , user_pseudo_id
  , event_name
  , event_timestamp
  , (select value.string_value from unnest(event_params) where key = 'page_location') as page_location
  , (select value.string_value from unnest(event_params) where key = 'page_title') as page_title  
  , items.item_id
  , items.item_name
  , items.item_brand
  , items.item_variant
  , items.item_category
  , items.price
  , items.quantity
  , IF(items.item_revenue IS NULL,0,items.item_revenue) as item_revenue
  , items.item_list_index
  , items.promotion_name  
  , ecommerce.total_item_quantity as ecommerce_total_item_quantity
  , ecommerce.purchase_revenue_in_usd as ecommerce_purchase_revenue_in_usd
  , ecommerce.purchase_revenue as ecommerce_purchase_revenue
  , ecommerce.shipping_value as ecommerce_shipping_value
  , ecommerce.tax_value as ecommerce_tax_value
  , ecommerce.unique_items as ecommerce_unique_items
  , ecommerce.transaction_id as ecommerce_transaction_id 
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
  ,(select value.string_value from unnest(event_params) where key = 'session_engaged') as session_engaged
  ,(select value.int_value from unnest(event_params) where key = 'engagement_time_msec') as engagement_time_msec

FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` ,
UNNEST (items) as items

WHERE _TABLE_SUFFIX BETWEEN format_date('%Y%m%d',start_date) AND format_date('%Y%m%d',end_date)