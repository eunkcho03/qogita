from dataclasses import dataclass
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

@dataclass
class TextConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    pooling_strategy: str = "mean"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TextProcessor:
    def __init__(self, config: TextConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(config.device)
        self.model.eval()
        self.uncased = "uncased" in config.model_name.lower()

    def transform_df(self, df: pd.DataFrame, title_col="Name", brand_col="Brand"):
        texts = [self.build_text(t, b) for t, b in zip(df[title_col], df[brand_col])]
        return self.embed_texts(texts)

    def tokenize_batches(self, texts):
        B = self.config.batch_size
        for i in range(0, len(texts), B):
            enc = self.tokenizer(
                texts[i:i+B],
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            yield {k: v.to(self.config.device) for k, v in enc.items()}

    @torch.no_grad()
    def embed_texts(self, texts):
        reps = []
        for enc in self.tokenize_batches(texts):
            out = self.model(**enc) 
            if self.config.pooling_strategy == "mean":
                emb = self.mean_pooling(out.last_hidden_state, enc["attention_mask"])
            else:  # "cls"
                emb = out.last_hidden_state[:, 0, :]
            reps.append(emb.cpu())
        return torch.cat(reps, dim=0)

    @torch.no_grad()
    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)              
        counts = mask.sum(dim=1).clamp(min=1e-9)              
        return summed / counts                                     

    def normalize_text(self, s):
        if not isinstance(s, str):
            s = ""
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        if self.uncased:
            s = s.lower()
        return s

    def build_text(self, title, brand):
        t = self.normalize_text(title)
        b = self.normalize_text(brand)
        return f"{t}. brand: {b}" if b else t
