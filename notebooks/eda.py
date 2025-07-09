import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

#loading the datasets
customers_df = pd.read_csv('../data/olist_customers_dataset.csv')
geolocation_df = pd.read_csv('../data/olist_geolocation_dataset.csv')
order_items_df = pd.read_csv('../data/olist_order_items_dataset.csv')
order_payments_df = pd.read_csv('../data/olist_order_payments_dataset.csv')
order_reviews_df = pd.read_csv('../data/olist_order_reviews_dataset.csv')
orders_df = pd.read_csv('../data/olist_orders_dataset.csv')
products_df = pd.read_csv('../data/olist_products_dataset.csv')
sellers_df = pd.read_csv('../data/olist_sellers_dataset.csv')
product_translation_df = pd.read_csv('../data/product_category_name_translation.csv')

print(customers_df.columns)
print(geolocation_df.columns)
print(order_items_df.columns)
print(order_payments_df.columns)
print(order_reviews_df.columns)
print(orders_df.columns)
print(products_df.columns)
print(sellers_df.columns)
print(product_translation_df.columns)

#Merging all the tables according to the relevance
updated_orders_payments_df=pd.merge(orders_df, order_payments_df, on = 'order_id', how ='left' )

updated_orders_payments_df=pd.merge(orders_df, order_payments_df, on = 'order_id', how ='left' )

updated_orders_payments_df=pd.merge(orders_df, order_payments_df, on = 'order_id', how ='left' )

updated_products_order_items_df=pd.merge(order_items_df, products_df, on='product_id', how='left' )

updated_orders_payment_reviews_df = pd.merge(updated_orders_payments_df, order_reviews_df, on='order_id', how='left')

# sellers_geolocation_df=pd.merge(sellers_df, geolocation_df, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')

# Take average lat/lng for each zip code prefix
geo_grouped = geolocation_df.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()

# Merge with sellers
sellers_geolocation_df = pd.merge(sellers_df, geo_grouped, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')

updated_sellers_geolocation_products_order_items_df= pd.merge(sellers_geolocation_df, updated_products_order_items_df, on='seller_id', how='right')

updated_orders_payment_reviews_customers_df=pd.merge(updated_orders_payment_reviews_df, customers_df, on='customer_id', how='left')

combined_df=pd.merge(updated_orders_payment_reviews_customers_df, updated_sellers_geolocation_products_order_items_df, on='order_id', how='left')

combined_df['order_status'].value_counts()

#Adding target Variable
combined_df['is_returned'] = combined_df['order_status'].apply(lambda x: 1 if x == 'canceled' else 0)

combined_df=combined_df[['customer_city', 'customer_state','seller_city', 'seller_state','price',
'product_category_name','product_weight_g','product_length_cm', 'product_height_cm', 'product_width_cm', 'product_photos_qty','freight_value', 'order_approved_at','order_delivered_customer_date', 'order_estimated_delivery_date', 'review_score','payment_value','payment_type', 'payment_installments','is_returned']]


final_df=pd.merge(combined_df,product_translation_df, on='product_category_name', how='left')

final_df.rename(columns={'product_category_name_english': 'product_category'}, inplace=True)

final_df=final_df[['customer_city', 'customer_state','seller_city', 'seller_state','price',
'product_category','product_weight_g','product_length_cm', 'product_height_cm', 'product_width_cm', 'product_photos_qty','freight_value', 'order_approved_at','order_delivered_customer_date', 'order_estimated_delivery_date', 'review_score','payment_value','payment_type', 'payment_installments','is_returned']]

#Converting Dates to datetime here
final_df['order_approved_at'] = pd.to_datetime(final_df['order_approved_at'])
final_df['order_delivered_customer_date'] = pd.to_datetime(final_df['order_delivered_customer_date'])
final_df['order_estimated_delivery_date'] = pd.to_datetime(final_df['order_estimated_delivery_date'])

#To understand delivery delay
final_df['delivery_delay'] = (
    final_df['order_delivered_customer_date'] - final_df['order_estimated_delivery_date']).dt.days.fillna(0)

#According to product volume
final_df['product_volume'] = final_df['product_length_cm'] * final_df['product_height_cm'] * final_df['product_width_cm']

#Dropping the used rows
final_df = final_df.drop(columns=[
    'order_approved_at', 'order_delivered_customer_date', 'order_estimated_delivery_date'
])

final_df = pd.get_dummies(final_df, columns=['payment_type', 'product_category', 'customer_state', 'seller_state'], drop_first=True)

final_df = final_df.dropna()

final_df = final_df.reset_index(drop=True)

non_numeric = final_df.select_dtypes(include='object').columns

final_df=final_df.drop(columns=['customer_city', 'seller_city'])

#Quick_look into data
print(final_df.shape)
print(final_df.info())

#checking for missing values
print(final_df.isna().sum().sort_values(ascending=False))

#understanding target variable
print(final_df['is_returned'].value_counts(normalize=True).plot(kind='bar', title='Return Rate'))

# As here this column is highly imbalanced, we will use stratified sampling or resampling later

print(final_df[['price', 'freight_value', 'review_score', 'delivery_delay']].describe())

print(final_df[['price', 'freight_value', 'delivery_delay']].hist(bins=30, figsize=(12, 6)))

# Price is highly skewed, have to consider using log-transforming or clipping outliers
# freight_value is right skewed, most shipping costs are under $100, can consider standardising or normalising
# Delivery_delay seems to be normally distributed, but centered around a negative mean, but this looks highly predictive, keep as it is
# In review_score, most are 4 or 5, can one-hot encode or bucket into groups

#Applying log transform on price 
#As this is feature enfgineering step we can do before splitting the data
final_df['log_price'] = np.log1p(final_df['price'])  # AS log1p handles 0 safely

# Bucket review_score
final_df['review_bucket'] = final_df['review_score'].apply(
    lambda x: 'low' if x <= 2 else 'medium' if x == 3 else 'high')

print(final_df[['log_price', 'freight_value', 'delivery_delay', 'review_bucket','review_score' ]].head(5))

print(final_df.corr(numeric_only=True)['is_returned'].sort_values(ascending=False))

print(final_df['payment_type_not_defined'].value_counts())

# for col in final_df.columns:
#     print(col)

# As with .corr(), we got the correlation of numeric values, we are using Cramers V for checking correlation with categorical values

final_df.to_csv("/Users/lakshmiprasannapoluru/Desktop/smart-return-predictor/data/final_df.csv", index=False)

#Finalising the features for base model
features = ['is_returned','delivery_delay','payment_value','price','customer_state_SP',
           'product_category_bed_bath_table','review_score','payment_type_not_defined' ]

#Setting feature and target columns
X = final_df[features].drop(columns=['is_returned'])
y = final_df['is_returned']