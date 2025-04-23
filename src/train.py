import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from utils import save_pickle

MODELS = {
    "logreg": LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced"),
    "svm": LinearSVC(class_weight="balanced"),
}

def build_pipe(clf):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True
                ),
            ),
            ("clf", clf),
        ]
    )

def run():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)  

    rows = []
    for name, clf in MODELS.items():
        pipe = build_pipe(clf)
        pipe.fit(pd.concat([train, val]).text, pd.concat([train, val]).label)
        save_pickle(pipe, f"outputs/models/{name}.pkl")

        preds = pipe.predict(test.text)
        rep = classification_report(
            test.label, preds, output_dict=True, zero_division=0
        )
        rows.append(
            {
                "model": name,
                "test_macro_f1": rep["macro avg"]["f1-score"],
                "test_accuracy": rep["accuracy"],
            }
        )

    pd.DataFrame(rows).to_csv("outputs/metrics/baselines.csv", index=False)

if __name__ == "__main__":
    run()