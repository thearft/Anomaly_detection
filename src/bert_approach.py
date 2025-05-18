import os
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from datasets import Dataset


def to_text(row: pd.Series) -> str:
    """
    Convert a DataFrame row of numeric features into a text description.
    """
    return (
        f"Time {int(row.ts)}: PID {int(row.PID)}, "
        f"{int(row.MINFLT)} minor faults, {int(row.MAJFLT)} major faults, "
        f"{row.MEM*100:.1f}% memory usage."
    )


def run_bert_approach(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fine-tune a small BERT model on labeled anomaly data and predict anomaly probabilities.

    Expects df to contain numeric columns:
      ['ts','PID','MINFLT','MAJFLT','VSTEXT','VSIZE','RSIZE','VGROW','RGROW','MEM']
    and a 'type' column with 0=Normal, 1=Anomaly.

    Returns a DataFrame for the test split with:
      - anomaly_score: probability of anomaly (class 1)
      - predicted_label: "Anomaly" or "Normal"
      - threshold: fixed 0.5
    """
    # Required columns
    features = ['ts','PID','MINFLT','MAJFLT','VSTEXT','VSIZE','RSIZE','VGROW','RGROW','MEM']
    missing = [c for c in features + ['type'] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Drop missing and reset index
    df_clean = df.dropna(subset=features + ['type']).reset_index(drop=True)
    df_clean['type'] = df_clean['type'].astype(int)

    # Build text column
    df_clean['text'] = df_clean.apply(to_text, axis=1)

    # Train/test split
    train_df, test_df = train_test_split(
        df_clean, test_size=0.2, stratify=df_clean['type'], random_state=42
    )

    # Build HF Dataset
    train_ds = Dataset.from_dict({
        'text': train_df['text'].tolist(),
        'label': train_df['type'].tolist()
    })
    test_ds = Dataset.from_dict({
        'text': test_df['text'].tolist(),
        'label': test_df['type'].tolist()
    })

    # Tokenizer & model
    model_name = 'prajjwal1/bert-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize
    def tokenize_fn(examples):
        tok = tokenizer(
            examples['text'], truncation=True, padding='max_length', max_length=128
        )
        tok['labels'] = examples['label']
        return tok

    tok_train = train_ds.map(tokenize_fn, batched=True)
    tok_test = test_ds.map(tokenize_fn, batched=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir='bert_toniot',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2
    )

    # Compute metrics for Trainer
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds)
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train & save
    trainer.train()
    trainer.save_model('bert_toniot')
    tokenizer.save_pretrained('bert_toniot')

    # Predict
    preds_output = trainer.predict(tok_test)
    probs = torch.softmax(
        torch.tensor(preds_output.predictions), dim=1
    )[:, 1].numpy()
    preds_binary = (probs >= 0.5).astype(int)

    # Print evaluation
    y_true = preds_output.label_ids
    print('Test Accuracy:', accuracy_score(y_true, preds_binary))
    print('Confusion Matrix:\n', confusion_matrix(y_true, preds_binary))
    print('Classification Report:\n', classification_report(y_true, preds_binary))

    # Build result DataFrame
    result = test_df.copy()
    result['anomaly_score'] = probs
    result['predicted_label'] = ['Anomaly' if p >= 0.5 else 'Normal' for p in probs]
    result['threshold'] = 0.5
    return result