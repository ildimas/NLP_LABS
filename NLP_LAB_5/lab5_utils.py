from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import evaluate

MAX_SOURCE_LENGTH = 512
DEFAULT_MODEL_NAME = "ai-forever/ruT5-base"

@dataclass
class TaskConfig:
    name: str
    target_field: str
    prefix: str
    max_target_length: int
    num_train_epochs: float = 3.0
    learning_rate: float = 3e-4
    batch_size: int = 4
    generation_max_length: int = 64
    num_beams: int = 4

def _build_preprocess_fn(
    tokenizer, prefix: str, target_field: str, max_target_length: int
):
    def preprocess_function(batch):
        inputs = [prefix + text.strip() for text in batch["text"]]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
        )

        labels = tokenizer(
            batch[target_field],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def tokenize_dataset(
    raw_dataset: DatasetDict,
    tokenizer,
    task_config: TaskConfig,
) -> DatasetDict:
    preprocess_function = _build_preprocess_fn(
        tokenizer=tokenizer,
        prefix=task_config.prefix,
        target_field=task_config.target_field,
        max_target_length=task_config.max_target_length,
    )
    return raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

def build_compute_metrics(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        if predictions.ndim > 2 or not np.issubdtype(predictions.dtype, np.integer):
            predictions = predictions.argmax(axis=-1)
        predictions = predictions.astype(np.int64, copy=False)
        vocab_size = tokenizer.vocab_size
        predictions = np.where(
            (predictions < 0) | (predictions >= vocab_size),
            tokenizer.pad_token_id,
            predictions,
        )
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics

def train_task(
    raw_dataset: DatasetDict,
    tokenizer,
    task: TaskConfig,
    output_root: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    seed: int = 42,
):
    set_seed(seed)
    output_dir = output_root / task.name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, task)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=task.learning_rate,
        num_train_epochs=task.num_train_epochs,
        per_device_train_batch_size=task.batch_size,
        per_device_eval_batch_size=task.batch_size,
        weight_decay=0.01,
        predict_with_generate=True,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
        generation_max_length=task.generation_max_length,
        generation_num_beams=task.num_beams,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        fp16=fp16,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {"train": train_result.metrics, "eval": eval_metrics},
            fp,
            ensure_ascii=False,
            indent=2,
        )

    return trainer, eval_metrics

def generate_sample(
    model_path: Path,
    prefix: str,
    text: str,
    max_length: int,
    num_beams: int = 4,
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prepared = prefix + text.strip()
    encoded = tokenizer(
        prepared,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True)