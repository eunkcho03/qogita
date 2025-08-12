import pandas as pd
import data_functions as df_funcs
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from text_processing import TextProcessor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from image_processing import ImageEmbedder
import torch
from tqdm import tqdm

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

print("Computing TEXT embeddings...", flush=True)
tp = TextProcessor()
train_text_emb = tp.transform_df(df_train)  
val_text_emb = tp.transform_df(df_val)
test_text_emb = tp.transform_df(df_test)
print("Done TEXT embeddings.", flush=True)

print("Computing IMAGE embeddings...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
IE = ImageEmbedder(device=device)
train_image_emb = IE.embed_dataframe(df_train, url_column="Image URL")
val_image_emb = IE.embed_dataframe(df_val, url_column="Image URL")          
test_image_emb = IE.embed_dataframe(df_test, url_column="Image URL")
print("Done IMAGE embeddings.", flush=True)
train_data = torch.cat([train_text_emb, train_image_emb], dim=1)
val_data = torch.cat([val_text_emb, val_image_emb], dim=1)
test_data = torch.cat([test_text_emb, test_image_emb], dim=1)

# Encode the categories
le = LabelEncoder().fit(df_train["Category"])
ytr = le.transform(df_train["Category"])
yva = le.transform(df_val["Category"])
yte = le.transform(df_test["Category"])
num_classes = len(le.classes_)
 
##############################################################
# Creat MLP
ytr_t = torch.tensor(ytr, dtype=torch.long)
yva_t = torch.tensor(yva, dtype=torch.long) 
yte_t = torch.tensor(yte, dtype=torch.long)

tr_ds = TensorDataset(train_data, ytr_t)
va_ds = TensorDataset(val_data, yva_t)
te_ds = TensorDataset(test_data, yte_t)

tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
va_dl = DataLoader(va_ds, batch_size=64, shuffle=False)
te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

input_dim = train_data.shape[1]

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

model = MLP(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


@torch.no_grad()
def evaluate(dataloader, model, criterion, desc="Eval"):
    model.eval()
    losses, all_logits, all_labels = [], [], []
    for xb, yb in tqdm(dataloader, desc=desc, leave=False):
        xb = xb.to(device).float()
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        all_logits.append(logits.detach().cpu())
        all_labels.append(yb.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds  = logits.argmax(dim=1)

    acc = accuracy_score(labels.numpy(), preds.numpy())
    macro_f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
    return (sum(losses) / max(1, len(losses))), {"acc": acc, "macro_f1": macro_f1}, preds


def train_epoch(train_loader, val_loader, model, optimizer, criterion):
    model.train()
    for xb, yb in tqdm(train_loader, desc="Train", leave=False):
        xb = xb.to(device).float()
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    train_loss, train_metrics, _ = evaluate(train_loader, model, criterion, desc="Eval(train)")
    _, val_metrics, vpred = evaluate(val_loader, model, criterion, desc="Eval(val)")
    return train_loss, train_metrics, val_metrics, vpred


best_val = -1.0
best_state = None
patience = 3
bad = 0
epochs = 15

for ep in range(1, epochs + 1):
    tr_loss, tr_m, va_m, _ = train_epoch(tr_dl, va_dl, model, optimizer, criterion)
    print(f"ep{ep:02d} | train acc={tr_m['acc']:.3f} f1={tr_m['macro_f1']:.3f} | "
          f"val acc={va_m['acc']:.3f} f1={va_m['macro_f1']:.3f}")
    if va_m["acc"] > best_val:
        best_val = va_m["acc"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping.")
            break

if best_state is not None:
    model.load_state_dict(best_state)


test_loss, test_metrics, test_preds = evaluate(te_dl, model, criterion)
print(f"TEST | loss={test_loss:.4f} acc={test_metrics['acc']:.3f} macro-F1={test_metrics['macro_f1']:.3f}")
print(classification_report(yte, test_preds.numpy(), target_names=le.classes_))