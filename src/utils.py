
import re, pickle, random, numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def clean(txt: str) -> str:
    txt = txt.lower().strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt

def save_pickle(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_metrics(pred):
    y_pred = pred.predictions.argmax(axis=-1)
    y_true = pred.label_ids
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": f1, "precision": p, "recall": r}

#def fix_seed(seed=42):
 #   random.seed(seed)
 #   np.random.seed(seed)
  #  torch.manual_seed(seed)
   # if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(seed)
