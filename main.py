import pandas as pd
import data_functions as df_funcs
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
# Read csv file
df = pd.read_csv("qogita_filtered_products.csv")

df.columns = df.columns.str.strip() 
category_counts = df["Category"].value_counts()

'''
There are categories with less than 10 products, which may not be suitable for training a model.
Therefore, I decided to filter out these categories. Less accurate but there's no other way tbh. 
'''

##############################################################
# Keep only categories with at least 10 products
df_filtered = df_funcs.filter_categories_with_min_products("qogita_filtered_products.csv", 10)

# Split the dataset into train, validation, and test sets
df_train, df_val, df_test = df_funcs.split_dataset(df_filtered, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

'''
# Save the filtered DataFrame to a new CSV file
df_train.to_csv("train.csv", index=False)
df_val.to_csv("validation.csv", index=False)
df_test.to_csv("test.csv", index=False)
'''
##############################################################



 