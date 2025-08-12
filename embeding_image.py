import os, pandas as pd, torch, hashlib, requests
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor as TPE
from sentence_transformers import SentenceTransformer

CLIP_MODEL = "clip-ViT-B-32"

def get_or_make_image_embeddings_clip(df, split, url_col="Image URL", cache_dir="models/artifacts", model_name=CLIP_MODEL, batch_size=256):
    os.makedirs(cache_dir, exist_ok=True); pt = os.path.join(cache_dir, f"image_emb_{split}_{model_name.replace('/','-')}.pt")
    if os.path.exists(pt):
        s = torch.load(pt, map_location="cpu"); return s["emb"] if isinstance(s, dict) else s
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.backends.cudnn.benchmark = True
    img_dir = Path(cache_dir)/"img_cache"/split; urls = df[url_col].astype(str).tolist()
    paths = [img_dir/f"{hashlib.sha1(u.encode()).hexdigest()[:16]}.jpg" for u in urls]; img_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(u,p):
        if p.exists(): return True
        try: r=requests.get(u,timeout=10); r.raise_for_status(); p.write_bytes(r.content); return True
        except: return False

    with TPE(max_workers=32) as ex:
        list(tqdm(ex.map(_fetch, urls, paths), total=len(urls), desc=f"Fetch {split} imgs"))

    images = [Image.open(p).convert("RGB") if p.exists() else Image.new("RGB",(224,224),(0,0,0)) for p in paths]
    model = SentenceTransformer(model_name, device=dev); use_cuda = (dev=="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
        emb = model.encode(images=images, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True).cpu()
    torch.save({"emb": emb}, pt); return emb

files = {"train":"train.csv","validation":"validation.csv","test":"test.csv"}
for split,f in files.items():
    if not os.path.exists(f): print(f"File {f} not found, skipping {split}"); continue
    df = pd.read_csv(f)
    if "Image URL" not in df.columns: raise KeyError(f"{f} missing 'Image URL'")
    e = get_or_make_image_embeddings_clip(df, split)
    print(f"[{split}] {e.shape} -> models/artifacts/image_emb_{split}_clip-ViT-B-32.pt")
