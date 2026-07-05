import argparse
import json
import os
import random
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from configs import get_cfg_defaults
from dataloader import DTIDataset, PreExtractedNpyDTIDataset
from domain_adaptator import Discriminator, ReverseLayerF
from model import MAGC_DTI


CLUSTER_COLUMNS = ("drug_cluster", "target_cluster")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MAGC-DTI with paper-aligned CDAN"
    )
    parser.add_argument(
        "--mode",
        choices=("default", "cross_domain"),
        default="default",
    )
    parser.add_argument(
        "--data_dir",
        default="../datasets/bindingdb/cluster",
        help="Directory containing CSV data and cluster annotations.",
    )
    parser.add_argument(
        "--use_pretrained_features",
        action="store_true",
        help=(
            "Use ChemBERTa/ESM-2 pooled NPY or token HDF5 features "
            "instead of raw graphs and protein indices."
        ),
    )
    parser.add_argument(
        "--feature_dir",
        default="../features/bindingdb",
        help="Default root directory for pre-extracted features.",
    )
    parser.add_argument("--source_feature_dir", default=None)
    parser.add_argument("--target_train_feature_dir", default=None)
    parser.add_argument("--validation_feature_dir", default=None)
    parser.add_argument("--target_test_feature_dir", default=None)

    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--source_split", default="source_train")
    parser.add_argument("--target_train_split", default="target_train")
    parser.add_argument("--validation_split", default="val")
    parser.add_argument("--target_test_split", default="target_test")

    parser.add_argument(
        "--train_csv",
        "--train_cluster_csv",
        dest="train_csv",
        default=None,
    )
    parser.add_argument(
        "--val_csv",
        "--val_cluster_csv",
        dest="val_csv",
        default=None,
    )
    parser.add_argument(
        "--test_csv",
        "--test_cluster_csv",
        dest="test_csv",
        default=None,
    )
    parser.add_argument(
        "--source_csv",
        "--source_cluster_csv",
        dest="source_csv",
        default=None,
    )
    parser.add_argument(
        "--target_train_csv",
        "--target_train_cluster_csv",
        dest="target_train_csv",
        default=None,
    )
    parser.add_argument(
        "--validation_csv",
        default=None,
        help=(
            "Held-out validation CSV. It is never used by the domain loss."
        ),
    )
    parser.add_argument(
        "--target_test_csv",
        "--target_test_cluster_csv",
        dest="target_test_csv",
        default=None,
    )
    parser.add_argument(
        "--validation_domain",
        choices=("source", "target"),
        default="source",
        help="Domain membership of the held-out validation split.",
    )

    parser.add_argument("--lambda_domain", type=float, default=0.1)
    parser.add_argument("--da_init_epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--analyze_clusters", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_csv(path, data_dir, split):
    resolved = Path(path) if path else Path(data_dir) / f"{split}.csv"
    if not resolved.is_file():
        raise FileNotFoundError(f"CSV file not found: {resolved}")
    return str(resolved)


def read_metadata(csv_path, require_clusters):
    frame = pd.read_csv(csv_path)
    required = {"SMILES", "Protein", "Y"}
    missing_data = required.difference(frame.columns)
    if missing_data:
        raise ValueError(
            f"{csv_path} is missing required columns: "
            f"{sorted(missing_data)}"
        )

    missing_clusters = set(CLUSTER_COLUMNS).difference(frame.columns)
    if require_clusters and missing_clusters:
        raise ValueError(
            f"{csv_path} is missing cluster columns required for the "
            f"paper cross-domain split: {sorted(missing_clusters)}"
        )
    return frame


class AnnotatedDTIDataset(Dataset):
    """Add validated cluster metadata to a raw or pre-extracted dataset."""

    def __init__(self, base_dataset, metadata, require_clusters=False):
        self.base_dataset = base_dataset
        self.metadata = metadata.reset_index(drop=True)
        if len(self.base_dataset) != len(self.metadata):
            raise ValueError(
                f"Feature rows ({len(self.base_dataset)}) and metadata rows "
                f"({len(self.metadata)}) do not match"
            )

        missing = set(CLUSTER_COLUMNS).difference(self.metadata.columns)
        if require_clusters and missing:
            raise ValueError(
                f"Cluster metadata is required; missing {sorted(missing)}"
            )
        self.has_clusters = not missing

        if hasattr(self.base_dataset, "_labels_full"):
            indices = getattr(
                self.base_dataset,
                "indices",
                np.arange(len(self.base_dataset)),
            )
            feature_labels = np.asarray(
                self.base_dataset._labels_full[indices],
                dtype=np.float32,
            )
            csv_labels = self.metadata["Y"].to_numpy(dtype=np.float32)
            if not np.allclose(feature_labels, csv_labels):
                raise ValueError(
                    "Feature labels and metadata CSV labels are not aligned"
                )

        self.drug_dim = getattr(self.base_dataset, "drug_dim", None)
        self.protein_dim = getattr(self.base_dataset, "protein_dim", None)

    def __len__(self):
        return len(self.base_dataset)

    def cluster_sets(self):
        if not self.has_clusters:
            return set(), set()
        return (
            set(self.metadata["drug_cluster"].tolist()),
            set(self.metadata["target_cluster"].tolist()),
        )

    def __getitem__(self, index):
        first, second, label = self.base_dataset[index]
        row = self.metadata.iloc[index]
        if self.has_clusters:
            drug_cluster = int(row["drug_cluster"])
            target_cluster = int(row["target_cluster"])
        else:
            drug_cluster = -1
            target_cluster = -1

        sample = {
            "label": float(label),
            "drug_cluster": drug_cluster,
            "target_cluster": target_cluster,
        }
        if hasattr(first, "ndata"):
            sample["graph"] = first
            sample["protein"] = second
        else:
            sample["drug_feat"] = first
            sample["protein_feat"] = second
        return sample


def build_dataset(
    csv_path,
    require_clusters,
    use_pretrained_features=False,
    feature_dir=None,
    split=None,
    max_drug_nodes=300,
):
    metadata = read_metadata(csv_path, require_clusters=require_clusters)
    if use_pretrained_features:
        if not feature_dir or not split:
            raise ValueError(
                "feature_dir and split are required in pretrained mode"
            )
        base_dataset = PreExtractedNpyDTIDataset(
            feature_dir=feature_dir,
            split=split,
            use_esm2=True,
            use_chembert=True,
        )
    else:
        base_dataset = DTIDataset(
            np.arange(len(metadata)),
            metadata,
            max_drug_nodes=max_drug_nodes,
        )
    return AnnotatedDTIDataset(
        base_dataset,
        metadata,
        require_clusters=require_clusters,
    )


def collate_batch(batch):
    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.float32,
    )
    drug_clusters = torch.tensor(
        [item["drug_cluster"] for item in batch],
        dtype=torch.long,
    )
    target_clusters = torch.tensor(
        [item["target_cluster"] for item in batch],
        dtype=torch.long,
    )
    result = {
        "label": labels,
        "drug_cluster": drug_clusters,
        "target_cluster": target_clusters,
    }

    if "graph" in batch[0]:
        result["graph"] = dgl.batch([item["graph"] for item in batch])
        result["protein"] = torch.as_tensor(
            np.asarray([item["protein"] for item in batch]),
            dtype=torch.long,
        )
        return result

    drug_features = [item["drug_feat"] for item in batch]
    protein_features = [item["protein_feat"] for item in batch]
    result["drug_feat"] = (
        torch.stack(drug_features)
        if drug_features[0].dim() == 1
        else pad_sequence(
            drug_features,
            batch_first=True,
            padding_value=0.0,
        )
    )
    result["protein_feat"] = (
        torch.stack(protein_features)
        if protein_features[0].dim() == 1
        else pad_sequence(
            protein_features,
            batch_first=True,
            padding_value=0.0,
        )
    )
    return result


def make_loader(dataset, batch_size, shuffle, num_workers):
    if shuffle and len(dataset) < 2:
        raise ValueError("A training dataset must contain at least 2 samples")
    drop_last = (
        shuffle
        and len(dataset) > batch_size
        and len(dataset) % batch_size == 1
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def validate_cross_domain_split(source, target_train, validation, target_test,
                                validation_domain):
    source_drugs, source_targets = source.cluster_sets()
    target_drugs, target_targets = target_train.cluster_sets()
    test_drugs, test_targets = target_test.cluster_sets()
    val_drugs, val_targets = validation.cluster_sets()

    target_drugs |= test_drugs
    target_targets |= test_targets
    if validation_domain == "source":
        source_drugs |= val_drugs
        source_targets |= val_targets
    else:
        target_drugs |= val_drugs
        target_targets |= val_targets

    for name, source_ids, target_ids in (
        ("drug", source_drugs, target_drugs),
        ("target", source_targets, target_targets),
    ):
        overlap = source_ids.intersection(target_ids)
        if overlap:
            preview = sorted(overlap, key=str)[:10]
            raise ValueError(
                f"{name} clusters overlap between source and target "
                f"domains: {preview}"
            )
        total = len(source_ids) + len(target_ids)
        if total == 0:
            raise ValueError(f"No {name} cluster IDs were found")
        source_ratio = len(source_ids) / total
        print(
            f"{name.capitalize()} clusters: source={len(source_ids)}, "
            f"target={len(target_ids)}, source_ratio={source_ratio:.3f}"
        )
        if not 0.50 <= source_ratio <= 0.70:
            raise ValueError(
                f"{name} source-cluster ratio {source_ratio:.3f} is not "
                "consistent with the paper's 60/40 split"
            )


def prediction_probabilities(logits):
    positive = torch.sigmoid(logits.reshape(-1))
    return torch.stack((1.0 - positive, positive), dim=1)


def multilinear_condition(features, logits):
    """Return flatten(f_joint outer_product g), exactly as paper Eq. (9)."""
    features = features.flatten(start_dim=1)
    probabilities = prediction_probabilities(logits)
    conditioned = torch.bmm(
        features.unsqueeze(2),
        probabilities.unsqueeze(1),
    ).flatten(start_dim=1)
    return conditioned, probabilities


def cdan_domain_loss(discriminator, features, logits, domain_labels, alpha):
    conditioned, _ = multilinear_condition(features, logits)
    reversed_features = ReverseLayerF.apply(conditioned, alpha)
    domain_logits = discriminator(reversed_features)
    return F.cross_entropy(
        domain_logits,
        domain_labels.long().reshape(-1),
    )


def forward_batch(model, batch, device):
    if "graph" in batch:
        drug_input = batch["graph"].to(device)
        protein_input = batch["protein"].to(device)
    else:
        drug_input = batch["drug_feat"].to(device)
        protein_input = batch["protein_feat"].to(device)
    _, _, joint_features, logits = model(
        drug_input,
        protein_input,
        mode="train",
    )
    return joint_features, logits.reshape(-1)


def safe_auroc(labels, probabilities):
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def compute_metrics(labels, probabilities, loss):
    labels = np.asarray(labels, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "loss": float(loss),
        "auroc": safe_auroc(labels, probabilities),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
    }


def grl_alpha(epoch, init_epoch, max_epochs):
    if epoch < init_epoch:
        return 0.0
    denominator = max(1, max_epochs - init_epoch - 1)
    progress = min(1.0, (epoch - init_epoch) / denominator)
    return float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


def train_epoch(model, loader, optimizer, task_criterion, device):
    model.train()
    total_loss = 0.0
    labels_all = []
    probabilities_all = []

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        labels = batch["label"].to(device)
        _, logits = forward_batch(model, batch, device)
        loss = task_criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        labels_all.extend(labels.detach().cpu().numpy())
        probabilities_all.extend(
            torch.sigmoid(logits).detach().cpu().numpy()
        )

    return compute_metrics(
        labels_all,
        probabilities_all,
        total_loss / len(loader),
    )


def train_domain_epoch(
    model,
    discriminator,
    source_loader,
    target_loader,
    optimizer,
    discriminator_optimizer,
    task_criterion,
    device,
    epoch,
    init_epoch,
    max_epochs,
    lambda_domain,
):
    model.train()
    discriminator.train()
    total_loss = 0.0
    total_domain_loss = 0.0
    labels_all = []
    probabilities_all = []
    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)
    n_batches = max(len(source_loader), len(target_loader))
    alpha = grl_alpha(epoch, init_epoch, max_epochs)

    for _ in range(n_batches):
        try:
            source_batch = next(source_iterator)
        except StopIteration:
            source_iterator = iter(source_loader)
            source_batch = next(source_iterator)
        try:
            target_batch = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_batch = next(target_iterator)

        optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        source_labels = source_batch["label"].to(device)

        source_features, source_logits = forward_batch(
            model,
            source_batch,
            device,
        )
        task_loss = task_criterion(source_logits, source_labels)

        if epoch >= init_epoch:
            target_features, target_logits = forward_batch(
                model,
                target_batch,
                device,
            )
            features = torch.cat(
                (source_features, target_features),
                dim=0,
            )
            logits = torch.cat(
                (source_logits, target_logits),
                dim=0,
            )
            domain_labels = torch.cat(
                (
                    torch.zeros(
                        source_features.size(0),
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.ones(
                        target_features.size(0),
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )
            domain_loss = cdan_domain_loss(
                discriminator,
                features,
                logits,
                domain_labels,
                alpha=alpha,
            )
            loss = task_loss + lambda_domain * domain_loss
            total_domain_loss += domain_loss.item()
        else:
            loss = task_loss

        loss.backward()
        optimizer.step()
        if epoch >= init_epoch:
            discriminator_optimizer.step()

        total_loss += loss.item()
        labels_all.extend(source_labels.detach().cpu().numpy())
        probabilities_all.extend(
            torch.sigmoid(source_logits).detach().cpu().numpy()
        )

    metrics = compute_metrics(
        labels_all,
        probabilities_all,
        total_loss / n_batches,
    )
    metrics["domain_loss"] = (
        total_domain_loss / n_batches if epoch >= init_epoch else 0.0
    )
    metrics["grl_alpha"] = alpha
    return metrics


@torch.no_grad()
def evaluate(model, loader, task_criterion, device):
    model.eval()
    total_loss = 0.0
    labels_all = []
    probabilities_all = []

    for batch in loader:
        labels = batch["label"].to(device)
        _, logits = forward_batch(model, batch, device)
        loss = task_criterion(logits, labels)
        total_loss += loss.item()
        labels_all.extend(labels.cpu().numpy())
        probabilities_all.extend(torch.sigmoid(logits).cpu().numpy())

    return compute_metrics(
        labels_all,
        probabilities_all,
        total_loss / len(loader),
    )


@torch.no_grad()
def analyze_cluster_performance(model, loader, device):
    model.eval()
    results = {"drug": {}, "target": {}}
    for batch in loader:
        labels = batch["label"].cpu().numpy()
        _, logits = forward_batch(model, batch, device)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        for kind, key in (
            ("drug", "drug_cluster"),
            ("target", "target_cluster"),
        ):
            for cluster_id, label, probability in zip(
                batch[key].numpy(),
                labels,
                probabilities,
            ):
                if int(cluster_id) < 0:
                    continue
                bucket = results[kind].setdefault(
                    int(cluster_id),
                    {"labels": [], "probabilities": []},
                )
                bucket["labels"].append(float(label))
                bucket["probabilities"].append(float(probability))

    summary = {"drug": {}, "target": {}}
    for kind, clusters in results.items():
        for cluster_id, values in clusters.items():
            labels = np.asarray(values["labels"])
            probabilities = np.asarray(values["probabilities"])
            summary[kind][cluster_id] = {
                "count": int(labels.size),
                "auroc": safe_auroc(labels, probabilities),
                "accuracy": float(
                    accuracy_score(
                        labels,
                        (probabilities >= 0.5).astype(np.int64),
                    )
                ),
            }
    return summary


def build_datasets(args, max_drug_nodes):
    use_features = args.use_pretrained_features
    if args.mode == "cross_domain":
        source_csv = resolve_csv(
            args.source_csv,
            args.data_dir,
            args.source_split,
        )
        target_train_csv = resolve_csv(
            args.target_train_csv,
            args.data_dir,
            args.target_train_split,
        )
        validation_csv = resolve_csv(
            args.validation_csv,
            args.data_dir,
            args.validation_split,
        )
        target_test_csv = resolve_csv(
            args.target_test_csv,
            args.data_dir,
            args.target_test_split,
        )
        source = build_dataset(
            source_csv,
            True,
            use_pretrained_features=use_features,
            feature_dir=args.source_feature_dir or args.feature_dir,
            split=args.source_split,
            max_drug_nodes=max_drug_nodes,
        )
        target_train = build_dataset(
            target_train_csv,
            True,
            use_pretrained_features=use_features,
            feature_dir=args.target_train_feature_dir or args.feature_dir,
            split=args.target_train_split,
            max_drug_nodes=max_drug_nodes,
        )
        validation = build_dataset(
            validation_csv,
            True,
            use_pretrained_features=use_features,
            feature_dir=args.validation_feature_dir or args.feature_dir,
            split=args.validation_split,
            max_drug_nodes=max_drug_nodes,
        )
        target_test = build_dataset(
            target_test_csv,
            True,
            use_pretrained_features=use_features,
            feature_dir=args.target_test_feature_dir or args.feature_dir,
            split=args.target_test_split,
            max_drug_nodes=max_drug_nodes,
        )
        validate_cross_domain_split(
            source,
            target_train,
            validation,
            target_test,
            args.validation_domain,
        )
        return source, validation, target_test, target_train

    train = build_dataset(
        resolve_csv(args.train_csv, args.data_dir, args.train_split),
        False,
        use_pretrained_features=use_features,
        feature_dir=args.feature_dir,
        split=args.train_split,
        max_drug_nodes=max_drug_nodes,
    )
    validation = build_dataset(
        resolve_csv(args.val_csv, args.data_dir, args.val_split),
        False,
        use_pretrained_features=use_features,
        feature_dir=args.feature_dir,
        split=args.val_split,
        max_drug_nodes=max_drug_nodes,
    )
    test = build_dataset(
        resolve_csv(args.test_csv, args.data_dir, args.test_split),
        False,
        use_pretrained_features=use_features,
        feature_dir=args.feature_dir,
        split=args.test_split,
        max_drug_nodes=max_drug_nodes,
    )
    return train, validation, test, None


def load_model_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def monitored_score(metrics, monitor):
    if monitor == "loss":
        return -metrics["loss"]
    if monitor == "auprc":
        return metrics["auprc"]
    return metrics["auroc"]


def main():
    args = parse_args()
    if args.lambda_domain < 0:
        raise ValueError("--lambda_domain must be non-negative")
    if args.da_init_epoch < 0:
        raise ValueError("--da_init_epoch must be non-negative")

    cfg = get_cfg_defaults()
    max_epochs = args.max_epochs or cfg.SOLVER.MAX_EPOCH
    batch_size = args.batch_size or cfg.SOLVER.BATCH_SIZE
    num_workers = (
        cfg.SOLVER.NUM_WORKERS
        if args.num_workers is None
        else args.num_workers
    )
    if args.da_init_epoch >= max_epochs and args.mode == "cross_domain":
        raise ValueError("--da_init_epoch must be smaller than max_epochs")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, validation_dataset, test_dataset, target_train_dataset = (
        build_datasets(args, cfg.DRUG.MAX_NODES)
    )

    cfg.PRETRAINED.USE_ESM2 = args.use_pretrained_features
    cfg.PRETRAINED.USE_CHEMBERT = args.use_pretrained_features
    cfg.PRETRAINED.FEATURE_DIR = args.feature_dir
    if args.use_pretrained_features:
        datasets = [train_dataset, validation_dataset, test_dataset]
        if target_train_dataset is not None:
            datasets.append(target_train_dataset)
        drug_dims = {dataset.drug_dim for dataset in datasets}
        protein_dims = {dataset.protein_dim for dataset in datasets}
        if (
            None in drug_dims
            or None in protein_dims
            or len(drug_dims) != 1
            or len(protein_dims) != 1
        ):
            raise ValueError(
                f"Inconsistent feature dimensions: drug={drug_dims}, "
                f"protein={protein_dims}"
            )
        cfg.PRETRAINED.CHEMBERT_DIM = drug_dims.pop()
        cfg.PRETRAINED.ESM2_DIM = protein_dims.pop()
        print(
            "Using pre-extracted features: "
            f"drug_dim={cfg.PRETRAINED.CHEMBERT_DIM}, "
            f"protein_dim={cfg.PRETRAINED.ESM2_DIM}"
        )

    train_loader = make_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    target_train_loader = (
        make_loader(
            target_train_dataset,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        if target_train_dataset is not None
        else None
    )

    model = MAGC_DTI(device=device, **cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.SOLVER.LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.SOLVER.LR_DECAY,
        patience=max(2, cfg.SOLVER.PATIENCE // 3),
    )
    task_criterion = nn.BCEWithLogitsLoss()

    discriminator = None
    discriminator_optimizer = None
    if args.mode == "cross_domain":
        domain_input_dim = cfg.DECODER.IN_DIM * 2
        discriminator = Discriminator(
            input_size=domain_input_dim,
        ).to(device)
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=cfg.SOLVER.LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / (
        "best_da_model.pth"
        if args.mode == "cross_domain"
        else "best_model.pth"
    )

    best_score = -float("inf")
    best_epoch = 0
    patience_counter = 0
    monitor = str(cfg.SOLVER.MONITOR).lower()
    history = []

    for epoch in range(max_epochs):
        if args.mode == "cross_domain":
            train_metrics = train_domain_epoch(
                model,
                discriminator,
                train_loader,
                target_train_loader,
                optimizer,
                discriminator_optimizer,
                task_criterion,
                device,
                epoch,
                args.da_init_epoch,
                max_epochs,
                args.lambda_domain,
            )
        else:
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                task_criterion,
                device,
            )

        validation_metrics = evaluate(
            model,
            validation_loader,
            task_criterion,
            device,
        )
        score = monitored_score(validation_metrics, monitor)
        if np.isfinite(score):
            scheduler.step(score)

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_AUROC={validation_metrics['auroc']:.4f} | "
            f"val_AUPRC={validation_metrics['auprc']:.4f}"
        )
        if args.mode == "cross_domain":
            print(
                f"  domain_loss={train_metrics['domain_loss']:.4f} | "
                f"grl_alpha={train_metrics['grl_alpha']:.4f}"
            )

        if np.isfinite(score) and (
            best_epoch == 0
            or score > best_score + cfg.SOLVER.MIN_DELTA
        ):
            best_score = score
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.SOLVER.PATIENCE:
                print(
                    f"Early stopping at epoch {epoch + 1}; "
                    f"best epoch was {best_epoch}."
                )
                break

    if best_epoch == 0:
        raise RuntimeError(
            f"Validation {monitor} was never finite; no model was saved"
        )
    model.load_state_dict(load_model_state(checkpoint_path, device))
    test_metrics = evaluate(model, test_loader, task_criterion, device)
    result = {
        "mode": args.mode,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "monitor": monitor,
        "best_score": best_score,
        "test": test_metrics,
        "history": history,
    }
    if args.analyze_clusters:
        result["cluster_metrics"] = analyze_cluster_performance(
            model,
            test_loader,
            device,
        )

    result_path = output_dir / "cdan_metrics.json"
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(f"Best epoch: {best_epoch}")
    print(json.dumps(test_metrics, indent=2, allow_nan=True))
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved metrics: {result_path}")


if __name__ == "__main__":
    main()
