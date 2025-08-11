import requests
import csv
import numpy as np
from io import StringIO
import pandas as pd
from functions import prepare_product_dataset_from_df

# Base URL for Qogita's API.
QOGITA_API_URL = "https://api.qogita.com"

# Login details for user.
QOGITA_EMAIL = "eunk0923.cho@gmail.com"
QOGITA_PASSWORD = "whdmsrud2003"

# Authentication request.
response = requests.post(url=f"{QOGITA_API_URL}/auth/login/",
                         json={"email": QOGITA_EMAIL, "password": QOGITA_PASSWORD}).json()

# Retrieve the access token and create the auth header to use in all requests.
access_token = response["accessToken"]
headers = {"Authorization": f"Bearer {access_token}"}

# Retrieve the active Cart identifier so that you can interact with the cart.
cart_qid = response["user"]["activeCartQid"]

##########################################################################################################
'''
# Assume auth was successful, and you have headers = { "Authorization": "Bearer <token>" }.

# Request to our catalog download endpoint.
response = requests.get(url=f"{QOGITA_API_URL}/variants/search/download/?"
                            f"&category_name=fragrance" # Filter by category name or URL slug
                            f"&brand_name=Paco Rabanne" # Filter by multiple brand names
                            f"&brand_name=Calvin Klein"
                            f"&stock_availability=in_stock" # Filter by products that are currently in stock
                            f"&page=1"
                            f"&size=10",
                        headers=headers).content.decode('utf-8')

# Create a CSV reader.
csv_reader = csv.reader(StringIO(response))

# Read the header row first.
headers = next(csv_reader)

# Now read the data rows line by line.
for row in csv_reader:
    print(row)
'''
##########################################################################################################

csv_text = requests.get(
    url=(
        f"{QOGITA_API_URL}/variants/search/download/?"
        f"&stock_availability=in_stock"
        f"&page=1"
        f"&size=10"           
    ),
    headers=headers).content.decode("utf-8")


df = pd.read_csv(StringIO(csv_text))

df_filtered = df[["Name", "Brand", "Category", "Image URL"]]

# save the filtered DataFrame to a CSV file
df_filtered.to_csv("qogita_filtered_products.csv", index=False)

