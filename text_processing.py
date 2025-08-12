import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

class TextProcessor:
    def __init__(self,
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 max_length=48,
                 batch_size=256,
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    def transform_df(self, df: pd.DataFrame, title_col="Name", brand_col=None):
        # simplest text: title only (optionally add brand if you want)
        if brand_col is None:
            texts = [str(t).strip() if isinstance(t, str) else "" for t in df[title_col]]
        else:
            texts = [
                (f"{str(t).strip()}. brand: {str(b).strip()}" if str(b).strip() else str(t).strip())
                for t, b in zip(df[title_col], df[brand_col])
            ]
        return self._embed_texts(texts)

    def _tokenize_batch(self, texts):
        return self.tokenizer(
            texts,
            padding=True,            # pad to longest in this batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def _embed_texts(self, texts):
        reps = []
        B = self.batch_size
        with torch.inference_mode():
            for i in range(0, len(texts), B):
                enc = self._tokenize_batch(texts[i:i+B]).to(self.device)
                # CLS pooling (first token)
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        out = self.model(**enc).last_hidden_state[:, 0, :]
                else:
                    out = self.model(**enc).last_hidden_state[:, 0, :]
                reps.append(out.detach().cpu())
        return torch.cat(reps, dim=0)  # [N, hidden_size]
