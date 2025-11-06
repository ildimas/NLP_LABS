#!/usr/bin/env python3
"""
Train a BiLSTM NER model on top of Russian ELMo embeddings.

The script downloads input data, performs BIO relabeling, trains a model, evaluates it
and finally runs inference on a custom list of external texts (e.g. snippets from media).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from razdel import tokenize as razdel_tokenize
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def biolu2bio(tags: Sequence[str]) -> List[str]:
    """Convert BIOLU tags to BIO format (replicates utils.relabeling.biolu2bio)."""
    converted = []
    for tag in tags:
        prefix = tag.split("-")[0]
        label = tag.split("-")[-1]
        if prefix == "U":
            converted.append(f"B-{label}")
        elif prefix == "L":
            converted.append(f"I-{label}")
        else:
            converted.append(tag)
    return converted


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ElmoSentenceEmbedder:
    """Wrapper around AllenNLP's Elmo module to obtain contextual embeddings."""

    def __init__(self, options_path: Path, weight_path: Path, device: torch.device) -> None:
        self.device = device
        self.elmo = Elmo(
            options_file=str(options_path),
            weight_file=str(weight_path),
            num_output_representations=1,
            dropout=0.0,
        ).to(self.device)
        self.elmo.eval()

    @torch.no_grad()
    def embed(self, tokens: Sequence[str]) -> torch.Tensor:
        character_ids = batch_to_ids([list(tokens)]).to(self.device)
        elmo_out = self.elmo(character_ids)
        # elmo_representations is a list with length == num_output_representations (1 here)
        embeddings = elmo_out["elmo_representations"][0][0]  # shape: (seq_len, emb_dim)
        return embeddings.cpu()


class ElmoSequenceDataset(Dataset):
    """Dataset that lazily caches sentence-level ELMo embeddings."""
    def __init__(
        self,
        tokens: Sequence[Sequence[str]],
        tags: Sequence[Sequence[str]],
        embedder: ElmoSentenceEmbedder,
        tag2idx: Dict[str, int],
    ) -> None:
        self.tokens = tokens
        self.tags = tags
        self.embedder = embedder
        self.tag2idx = tag2idx
        self._cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.tokens)

    def _embed(self, idx: int) -> torch.Tensor:
        if idx not in self._cache:
            sentence = list(self.tokens[idx])
            embedding = self.embedder.embed(sentence)
            self._cache[idx] = embedding.clone().detach()
        return self._cache[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        embeddings = self._embed(idx)
        tag_ids = torch.tensor(
            [self.tag2idx[tag] for tag in self.tags[idx]], dtype=torch.long
        )
        length = embeddings.size(0)
        return embeddings, tag_ids, length


def collate_batch(
    batch: Iterable[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings, tags, lengths = zip(*batch)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    padded_embeddings = pad_sequence(embeddings, batch_first=True)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=-100)
    return padded_embeddings, padded_tags, lengths_tensor


class BiLstmTagger(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        num_labels: int,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return logits


@dataclass
class Metrics:
    loss: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(loader, desc="Train", leave=False)
    for embeddings, tags, lengths in iterator:
        embeddings = embeddings.to(device)
        tags = tags.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(embeddings, lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        valid_tokens = (tags != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
        iterator.set_postfix({"loss": loss.item()})

    return total_loss / max(total_tokens, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    idx2tag: Dict[int, str],
    device: torch.device,
    stage: str = "Eval",
) -> Tuple[Metrics, List[List[str]], List[List[str]]]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    gold_sequences: List[List[str]] = []
    pred_sequences: List[List[str]] = []

    iterator = tqdm(loader, desc=stage, leave=False)
    with torch.no_grad():
        for embeddings, tags, lengths in iterator:
            embeddings = embeddings.to(device)
            tags = tags.to(device)
            logits = model(embeddings, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1))

            mask = tags != -100
            token_count = mask.sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count

            preds = logits.argmax(dim=-1).cpu().numpy()
            gold = tags.cpu().numpy()
            lengths_np = lengths.numpy()

            for pred_seq, gold_seq, length in zip(preds, gold, lengths_np):
                pred_labels = [idx2tag[idx] for idx in pred_seq[:length]]
                gold_labels = [idx2tag[idx] for idx in gold_seq[:length]]
                pred_sequences.append(pred_labels)
                gold_sequences.append(gold_labels)

            iterator.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / max(total_tokens, 1)
    precision = precision_score(gold_sequences, pred_sequences, zero_division=0)
    recall = recall_score(gold_sequences, pred_sequences, zero_division=0)
    f1 = f1_score(gold_sequences, pred_sequences, zero_division=0)
    metrics = Metrics(loss=avg_loss, precision=precision, recall=recall, f1=f1)
    return metrics, gold_sequences, pred_sequences


def load_dataset(pickle_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    df = pd.read_pickle(pickle_path)
    tokens = df["tokens"].tolist()
    tags = df["ner_tags"].tolist()
    tags = [biolu2bio(seq) for seq in tags]
    return tokens, tags


def build_tag_mappings(tags: Sequence[Sequence[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    tag_set = sorted({tag for seq in tags for tag in seq})
    tag2idx = {tag: idx for idx, tag in enumerate(tag_set)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return tag2idx, idx2tag


def extract_entities(tokens: Sequence[str], tags: Sequence[str]) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    buffer: List[str] = []
    current_label: str | None = None

    for token, tag in zip(tokens, tags):
        if tag == "O":
            if buffer:
                entities.append((" ".join(buffer), current_label or ""))
                buffer = []
                current_label = None
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "B":
            if buffer:
                entities.append((" ".join(buffer), current_label or label))
            buffer = [token]
            current_label = label
        elif prefix == "I":
            if buffer and current_label == label:
                buffer.append(token)
            else:
                if buffer:
                    entities.append((" ".join(buffer), current_label or label))
                buffer = [token]
                current_label = label
        else:
            # Gracefully handle unexpected prefixes
            if buffer:
                entities.append((" ".join(buffer), current_label or label))
            buffer = [token]
            current_label = label

    if buffer:
        entities.append((" ".join(buffer), current_label or ""))
    return entities


def tokenize_text(text: str) -> List[str]:
    return [token.text for token in razdel_tokenize(text)]


def predict_tags_for_sentence(
    model: nn.Module,
    embedder: ElmoSentenceEmbedder,
    tokens: Sequence[str],
    idx2tag: Dict[int, str],
    device: torch.device,
) -> List[str]:
    embeddings = embedder.embed(tokens)
    tensor = embeddings.unsqueeze(0).to(device)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(tensor, lengths)
        pred_ids = logits.argmax(dim=-1)[0][: len(tokens)].cpu().tolist()
    return [idx2tag[idx] for idx in pred_ids]


def read_external_texts(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in raw if line.strip()]


def log_external_predictions(
    model: nn.Module,
    embedder: ElmoSentenceEmbedder,
    texts: Sequence[str],
    idx2tag: Dict[int, str],
    device: torch.device,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for text in texts:
        tokens = tokenize_text(text)
        if not tokens:
            results.append({"text": text, "tokens": [], "predicted_tags": [], "entities": []})
            continue
        pred_tags = predict_tags_for_sentence(model, embedder, tokens, idx2tag, device)
        entities = extract_entities(tokens, pred_tags)
        results.append(
            {
                "text": text,
                "tokens": tokens,
                "predicted_tags": pred_tags,
                "entities": entities,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ELMo + BiLSTM NER on Detailed-NER-Dataset-RU")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("../data/Detailed-NER-Dataset-RU/dataset/detailed-ner_dataset-ru.pickle"),
        help="Path to the Detailed-NER-Dataset-RU pickle file.",
    )
    parser.add_argument(
        "--elmo-model-dir",
        type=Path,
        default=Path("../models/ruwikiruscorpora_tokens_elmo_1024_2019"),
        help="Path to the directory with ELMo model files.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=12, help="Mini-batch size.")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size of the BiLSTM.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of dataset reserved for test.")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of the training split used for validation (relative to train split).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--external-texts",
        type=Path,
        default=Path("external_texts_rbc.txt"),
        help="Path to file with external texts (one per line) for qualitative evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where metrics and predictions will be saved.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of LSTM layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability applied after LSTM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print("=== Configuration ===")
    print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2, ensure_ascii=False))

    print("\n=== Loading dataset ===")
    tokens, tags = load_dataset(args.dataset_path)
    print(f"Loaded {len(tokens)} sentences.")

    indices = np.arange(len(tokens))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=args.val_size, random_state=args.seed, shuffle=True
    )

    def select(items: Sequence, idxs: Sequence[int]) -> List:
        return [items[i] for i in idxs]

    tokens_train, tags_train = select(tokens, train_idx), select(tags, train_idx)
    tokens_val, tags_val = select(tokens, val_idx), select(tags, val_idx)
    tokens_test, tags_test = select(tokens, test_idx), select(tags, test_idx)

    print(
        f"Split sizes -> train: {len(tokens_train)}, val: {len(tokens_val)}, test: {len(tokens_test)}"
    )

    tag2idx, idx2tag = build_tag_mappings(tags)
    print(f"Detected {len(tag2idx)} distinct BIO tags: {sorted(tag2idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n=== Loading ELMo model ===")
    options_path = args.elmo_model_dir / "options.json"
    weight_path = args.elmo_model_dir / "model.hdf5"
    if not options_path.exists() or not weight_path.exists():
        raise FileNotFoundError(
            f"ELMo model files not found in {args.elmo_model_dir}. "
            f"Expected options.json and model.hdf5."
        )
    elmo_embedder = ElmoSentenceEmbedder(options_path=options_path, weight_path=weight_path, device=device)

    train_dataset = ElmoSequenceDataset(tokens_train, tags_train, elmo_embedder, tag2idx)
    val_dataset = ElmoSequenceDataset(tokens_val, tags_val, elmo_embedder, tag2idx)
    test_dataset = ElmoSequenceDataset(tokens_test, tags_test, elmo_embedder, tag2idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0
    )

    model = BiLstmTagger(
        embedding_dim=1024,
        hidden_size=args.hidden_size,
        num_labels=len(tag2idx),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    history = []
    best_state = None
    best_val_f1 = -1.0

    print("\n=== Training ===")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, idx2tag, device, stage="Validation")

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.loss,
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "val_f1": val_metrics.f1,
        }
        history.append(epoch_summary)
        print(
            f"Epoch {epoch} -> train_loss: {train_loss:.4f}, "
            f"val_loss: {val_metrics.loss:.4f}, "
            f"val_precision: {val_metrics.precision:.4f}, "
            f"val_recall: {val_metrics.recall:.4f}, "
            f"val_f1: {val_metrics.f1:.4f}"
        )

        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics.to_dict(),
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        print(
            f"\nLoaded best model from epoch {best_state['epoch']} "
            f"(val_f1={best_state['val_metrics']['f1']:.4f})."
        )

    print("\n=== Test evaluation ===")
    test_metrics, gold_sequences, pred_sequences = evaluate(
        model, test_loader, criterion, idx2tag, device, stage="Test"
    )
    print(
        f"Test loss: {test_metrics.loss:.4f}, precision: {test_metrics.precision:.4f}, "
        f"recall: {test_metrics.recall:.4f}, f1: {test_metrics.f1:.4f}"
    )
    report_text = classification_report(gold_sequences, pred_sequences, zero_division=0)
    print("\nSeqeval classification report:\n")
    print(report_text)

    metrics_payload = {
        "history": history,
        "best_val": best_state["val_metrics"] if best_state else None,
        "test": test_metrics.to_dict(),
        "classification_report": report_text,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved metrics to {metrics_path}")

    if args.external_texts.exists():
        print("\n=== External texts inference ===")
        external_texts = read_external_texts(args.external_texts)
        results = log_external_predictions(model, elmo_embedder, external_texts, idx2tag, device)
        for item in results:
            print("\nText:", item["text"])
            print("Entities:", item["entities"] if item["entities"] else "—")
        external_path = args.output_dir / "external_predictions.json"
        external_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved external predictions to {external_path}")
    else:
        print("\nNo external texts file found; skipping qualitative evaluation.")

    total_time = time.time() - t_start
    print(f"\nDone in {total_time/60:.2f} minutes.")


if __name__ == "__main__":
    main()
