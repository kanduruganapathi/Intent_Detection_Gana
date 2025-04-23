

import argparse, os, pandas as pd
from sklearn.model_selection import train_test_split
from utils import clean
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import clean 

def main():
    csv_path = "H:\Data science\Gen AI\Intent_Detection\data\Gana_Train.csv"  
    df = pd.read_csv(csv_path)
    if {"text", "label"} - set(df.columns):
        raise ValueError("CSV must contain columns named 'text' and 'label'")
    df["text"] = df["text"].astype(str).apply(clean)
   
    train, temp = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["label"], random_state=42
    )
    
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)


if __name__ == "__main__":
    main()  