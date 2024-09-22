DECLARE start_date DATE DEFAULT '2021-01-01';
DECLARE end_date DATE DEFAULT '2021-12-31';



WITH prep as (
SELECT
  user_pseudo_id
, session_id
, event_timestamp
, event_name
, item_id
, CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END as purchase_user

, CASE WHEN event_name = 'view_item' THEN event_timestamp END as view_item
, CASE WHEN event_name = 'view_item' THEN session_id END as view_item_session

, CASE WHEN event_name = 'add_to_cart' THEN session_id END as add_to_cart
, CASE WHEN event_name = 'begin_checkout' THEN session_id END as begin_checkout
, CASE WHEN event_name = 'add_payment_info' THEN session_id END as add_payment_info
, CASE WHEN event_name = 'purchase' THEN session_id END as purchase


FROM ist-data-science.ga4_google_merch_store.flattened_raw
WHERE event_date BETWEEN start_date and end_date

)


SELECT
  user_pseudo_id,
  MAX(purchase_user) as purchase_user
, COUNT(DISTINCT session_id) as nr_sessions
, COUNT(DISTINCT view_item) as items_viewed
, COUNT(DISTINCT view_item_session) as view_item_sessions

, COUNT(DISTINCT add_to_cart) as add_to_cart_sessions
, COUNT(DISTINCT begin_checkout) as begin_checkout_sessions
, COUNT(DISTINCT add_payment_info) as add_payment_info_sessions
, COUNT(DISTINCT purchase) as purchase_sessions

FROM prep
GROUP BY user_pseudo_id
