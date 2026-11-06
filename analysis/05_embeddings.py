"""Extract frozen BERT-base-chinese embeddings (CLS + mean pooled) for all records."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json, hashlib
import numpy as np, pandas as pd, torch
sys.stdout.reconfigure(encoding="utf-8")

from provenance import corpus
df = corpus.load().reset_index(drop=True)
texts = df.abstract_clean.tolist()

from transformers import AutoTokenizer, AutoModel
model_id = os.environ.get("BERT_MODEL", "bert-base-chinese")
revision = os.environ.get("BERT_REVISION", "8f23c25b06e129b6c986331a13d8d025a92cf0ea")
local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
tok = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=local_only)
model = AutoModel.from_pretrained(model_id, revision=revision, local_files_only=local_only).eval()
torch.set_num_threads(int(os.environ.get("BERT_THREADS", "4")))

BS, MAXL = int(os.environ.get("BERT_BATCH_SIZE", "16")), 384
cls_out, mean_out = [], []
with torch.no_grad():
    for i in range(0, len(texts), BS):
        batch = texts[i:i + BS]
        enc = tok(batch, padding=True, truncation=True, max_length=MAXL, return_tensors="pt")
        h = model(**enc).last_hidden_state            # B,T,H
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        cls_out.append(h[:, 0].numpy())
        mean_out.append(mean.numpy())
        if i % (BS * 10) == 0:
            print(i, "/", len(texts), flush=True)

np.save(PROCESSED_DIR / "emb_cls.npy", np.vstack(cls_out).astype(np.float32))
np.save(PROCESSED_DIR / "emb_mean.npy", np.vstack(mean_out).astype(np.float32))
metadata = dict(model=model_id, revision=revision, max_length=MAXL,
                pooling="attention-mask mean, including special tokens",
                records=len(df), corpus_sha256=hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
                record_ids_sha256=hashlib.sha256("\n".join(df.record_id).encode()).hexdigest())
(PROCESSED_DIR / "embeddings_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("done", np.vstack(cls_out).shape)
