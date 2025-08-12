import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

df_train = pd.read_csv("train2.csv")
df_val   = pd.read_csv("validation2.csv")
df_test  = pd.read_csv("test2.csv")

Xtr_txt = torch.load("models/artifacts/text/text_cash/text_emb_train.pt", map_location="cpu", weights_only=True)
Xva_txt = torch.load("models/artifacts/text/text_cash/text_emb_validation.pt", map_location="cpu", weights_only=True)
Xte_txt = torch.load("models/artifacts/text/text_cash/text_emb_test.pt", map_location="cpu", weights_only=True)

train_data, val_data, test_data = Xtr_txt, Xva_txt, Xte_txt

all_cats = pd.concat([df_train["Category"], df_val["Category"], df_test["Category"]], ignore_index=True)
le  = LabelEncoder().fit(all_cats)
ytr = torch.tensor(le.transform(df_train["Category"]), dtype=torch.long)
yva = torch.tensor(le.transform(df_val["Category"]),   dtype=torch.long)
yte = torch.tensor(le.transform(df_test["Category"]),  dtype=torch.long)
num_classes = len(le.classes_)

tr_ds = TensorDataset(train_data, ytr)
va_ds = TensorDataset(val_data,   yva)
te_ds = TensorDataset(test_data,  yte)

tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
va_dl = DataLoader(va_ds, batch_size=64, shuffle=False)
te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = train_data.shape[1]

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # increased from 512 → 1024
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x)); x = self.dropout(x)
        x = self.relu(self.fc2(x)); x = self.dropout(x)
        x = self.relu(self.fc3(x)); x = self.dropout(x)
        x = self.relu(self.fc4(x)); x = self.dropout(x)
        return self.fc5(x)
      
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
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    train_loss, train_metrics, _ = evaluate(train_loader, model, criterion, desc="Eval(train)")
    _, val_metrics, _ = evaluate(val_loader, model, criterion, desc="Eval(val)")
    return train_loss, train_metrics, val_metrics

best_val, best_state = -1.0, None
patience, bad, epochs = 3, 0, 50

for ep in range(1, epochs + 1):
    tr_loss, tr_m, va_m = train_epoch(tr_dl, va_dl, model, optimizer, criterion)
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
print(classification_report(
    yte.numpy(),
    test_preds.numpy(),
    labels=list(range(num_classes)),
    target_names=le.classes_[:num_classes]
))
