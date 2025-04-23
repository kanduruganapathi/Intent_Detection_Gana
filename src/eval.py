
import pandas as pd, os


def main():
    base = pd.read_csv("outputs/metrics/baselines.csv")
    bert = pd.read_csv("outputs/metrics/bert_tf.csv")
    bert = bert.rename(
        columns={"eval_macro_f1": "test_macro_f1", "eval_accuracy": "test_accuracy"}
    ).assign(model="distilbert")
    final = pd.concat(
        [base[["model", "test_macro_f1", "test_accuracy"]],
         bert[["model", "test_macro_f1", "test_accuracy"]]],
        ignore_index=True,
    )
    final.to_csv("outputs/metrics/summary.csv", index=False)
    print(final)

if __name__ == "__main__":
    main()
