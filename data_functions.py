import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split

def filter_categories_with_min_products(file_path, min_products):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    category_counts = df["Category"].value_counts()
    categories_to_keep = category_counts[category_counts >= min_products].index
    df_filtered = df[df["Category"].isin(categories_to_keep)]
    return df_filtered

def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    df_train, df_temp = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state)
    val_size = val_ratio / (val_ratio + test_ratio)  # fraction of temp that goes to validation
    df_val, df_test = train_test_split(df_temp, test_size=(1 - val_size), random_state=random_state)

    return df_train, df_val, df_test

def normalize_text(s):
    if not isinstance(s, str):
        s = ""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)  # Replace multiple spaces with a single space
    s = s.lower()  # Convert to lowercase
    return s

def build_text(title, brand):
    t = normalize_text(title)
    b = normalize_text(brand)
    if t and b:
        return f"{t}. brand: {b}"
    else:
        return t
    

# when text encoder like BERT is used, it doesn't give you one vector but gives matrix but only one vector is needed. So we just use mean pooling!

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts  
