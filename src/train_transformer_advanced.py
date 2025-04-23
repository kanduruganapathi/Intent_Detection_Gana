import os
import pandas as pd
import numpy as np
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def load_data():
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    return train_df, val_df, test_df

def encode_labels(train_df, val_df, test_df):
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])
    val_df['label'] = le.transform(val_df['label'])
    test_df['label'] = le.transform(test_df['label'])
    return le, train_df, val_df, test_df

def tokenize_texts(tokenizer, texts, max_len=64):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="tf"
    )

def prepare_dataset(df, tokenizer, max_len=64):
    encodings = tokenize_texts(tokenizer, df['text'], max_len=max_len)
    labels = to_categorical(df['label'])
    return encodings, labels

def build_model(model_name, num_labels):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    optimizer = Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df, val_df, test_df = load_data()
    le, train_df, val_df, test_df = encode_labels(train_df, val_df, test_df)
    num_labels = len(le.classes_)

    train_enc, train_labels = prepare_dataset(train_df, tokenizer)
    val_enc, val_labels = prepare_dataset(val_df, tokenizer)
    test_enc, test_labels = prepare_dataset(test_df, tokenizer)

    model = build_model(model_name, num_labels)

    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(
        x={'input_ids': train_enc['input_ids'], 'attention_mask': train_enc['attention_mask']},
        y=train_labels,
        validation_data=(
            {'input_ids': val_enc['input_ids'], 'attention_mask': val_enc['attention_mask']},
            val_labels
        ),
        epochs=6,
        batch_size=16,
        callbacks=[es]
    )

    preds = model.predict({'input_ids': test_enc['input_ids'], 'attention_mask': test_enc['attention_mask']})
    y_pred = np.argmax(preds.logits, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)

    metrics_df = pd.DataFrame({
        'model': [model_name],
        'test_macro_f1': [report['macro avg']['f1-score']],
        'test_accuracy': [report['accuracy']]
    })
 
    metrics_df.to_csv("outputs/metrics/bert_tf.csv", index=False)
    
    print(classification_report(y_true, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    os.makedirs("outputs/metrics", exist_ok=True)
    main()