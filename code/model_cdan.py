import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score
)
from model import MAGC_DTI
import dgl
from rdkit import Chem
import math
import warnings
from domain_adaptator import ReverseLayerF, Discriminator
import argparse


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = nn.ParameterList(
            [
                nn.Parameter(torch.randn(input_dim_list[i], output_dim), requires_grad=False)
                for i in range(self.input_num)
            ]
        )

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        return super(RandomLayer, self).cuda()


def prediction_probabilities(outputs):
    positive = torch.sigmoid(torch.clamp(outputs.view(-1, 1), min=-30.0, max=30.0))
    positive = torch.clamp(positive, min=1e-6, max=1.0 - 1e-6)
    return torch.cat([1.0 - positive, positive], dim=1)


def condition_domain_features(features, outputs, random_layer=None):
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    probabilities = prediction_probabilities(outputs)

    if random_layer is None:
        return features, probabilities

    if isinstance(random_layer, RandomLayer):
        return random_layer([features, probabilities]), probabilities
    elif isinstance(random_layer, nn.Linear):
        combined = torch.cat([features, probabilities], dim=1)
        return random_layer(combined), probabilities
    else:
        raise TypeError(f"Unsupported random_layer type: {type(random_layer)}")


def apply_random_layer(random_layer, features, outputs):
    domain_features, _ = condition_domain_features(features, outputs, random_layer)
    return domain_features


def cdan_domain_loss(discriminator, features, outputs, domain_labels, alpha=1.0, random_layer=None, use_entropy=True):
    domain_features, probabilities = condition_domain_features(features, outputs, random_layer)
    domain_features = ReverseLayerF.apply(domain_features, alpha)
    domain_logits = discriminator(domain_features)
    domain_labels = domain_labels.long().view(-1)

    losses = F.cross_entropy(domain_logits, domain_labels, reduction="none")
    if use_entropy:
        entropy = -torch.sum(probabilities.detach() * torch.log(probabilities.detach() + 1e-8), dim=1)
        weights = 1.0 + torch.exp(-entropy)
        losses = losses * (weights / weights.mean().clamp_min(1e-8))

    return losses.mean()


parser = argparse.ArgumentParser(description="MAGC-DTI with clustering and domain adaptation")
parser.add_argument(
    '--mode',
    default='default',
    choices=['default', 'cross_domain'],
    help='Experiment mode: default or cross_domain'
)
parser.add_argument(
    '--lambda_cluster',
    type=float,
    default=0.1,
    help='Weight of the clustering loss (default: 0.1)'
)
parser.add_argument(
    '--lambda_domain',
    type=float,
    default=0.1,
    help='Weight of the domain adaptation loss (default: 0.1)'
)
parser.add_argument(
    '--use_cluster_loss',
    action='store_true',
    help='Whether to use the cluster consistency loss'
)
parser.add_argument(
    '--analyze_clusters',
    action='store_true',
    help='Whether to analyze performance across different clusters'
)
parser.add_argument(
    '--feature_dir',
    type=str,
    default='../code/features/varlen',
    help='Root directory for pre-extracted features'
)
parser.add_argument(
    '--feature_strategy',
    type=str,
    default=None,
    help='Optional pooling strategy suffix for flat feature files, e.g. mean_mean'
)
parser.add_argument(
    '--source_feature_dir',
    type=str,
    default=None,
    help='Feature root for source-domain training data; defaults to --feature_dir'
)
parser.add_argument(
    '--target_train_feature_dir',
    type=str,
    default=None,
    help='Feature root for target-domain adaptation/validation data; defaults to --feature_dir'
)
parser.add_argument(
    '--target_test_feature_dir',
    type=str,
    default=None,
    help='Feature root for target-domain test data; defaults to --feature_dir'
)
parser.add_argument(
    '--train_split',
    type=str,
    default='train',
    help='Split name for default-mode training features'
)
parser.add_argument(
    '--val_split',
    type=str,
    default='val',
    help='Split name for default-mode validation features'
)
parser.add_argument(
    '--test_split',
    type=str,
    default='test',
    help='Split name for default-mode test features'
)
parser.add_argument(
    '--source_split',
    type=str,
    default='source_train',
    help='Split name for source-domain features in cross_domain mode'
)
parser.add_argument(
    '--target_train_split',
    type=str,
    default='target_train',
    help='Split name for target-domain training features in cross_domain mode'
)
parser.add_argument(
    '--target_test_split',
    type=str,
    default='target_test',
    help='Split name for target-domain test features in cross_domain mode'
)
parser.add_argument(
    '--train_cluster_csv',
    type=str,
    default=None,
    help='Optional CSV with drug_cluster/target_cluster columns for default train split'
)
parser.add_argument(
    '--val_cluster_csv',
    type=str,
    default=None,
    help='Optional CSV with drug_cluster/target_cluster columns for default validation split'
)
parser.add_argument(
    '--test_cluster_csv',
    type=str,
    default=None,
    help='Optional CSV with drug_cluster/target_cluster columns for default test split'
)
parser.add_argument(
    '--source_cluster_csv',
    type=str,
    default=None,
    help='Optional CSV with cluster columns for source-domain features'
)
parser.add_argument(
    '--target_train_cluster_csv',
    type=str,
    default=None,
    help='Optional CSV with cluster columns for target-domain train features'
)
parser.add_argument(
    '--target_test_cluster_csv',
    type=str,
    default=None,
    help='Optional CSV with cluster columns for target-domain test features'
)

warnings.filterwarnings('ignore')


class ClusterDTIDataset(Dataset):
    def __init__(self, csv_file, max_protein_len=1200):
        self.data = pd.read_csv(csv_file)
        self.max_protein_len = max_protein_len

    def __len__(self):
        return len(self.data)

    def _smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogen atoms
        mol = Chem.AddHs(mol)

        # Build richer atom features (75 dimensions)
        atom_feats = []
        for atom in mol.GetAtoms():
            # Atom type one-hot encoding
            atom_type = atom.GetAtomicNum()
            type_features = [0] * 43  # Support the first 43 elements
            if 0 < atom_type <= 43:
                type_features[atom_type - 1] = 1

            feat = []
            feat.extend(type_features)
            feat.append(int(atom.GetIsAromatic()))  # Aromaticity

            # Hybridization type
            hybridization_types = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ]
            hybridization = [0] * len(hybridization_types)
            hybridization_type = atom.GetHybridization()
            if hybridization_type in hybridization_types:
                hybridization[hybridization_types.index(hybridization_type)] = 1
            feat.extend(hybridization)

            # Additional atomic features
            feat.append(atom.GetFormalCharge())      # Formal charge
            feat.append(len(atom.GetNeighbors()))    # Degree
            feat.append(atom.GetTotalNumHs())        # Total number of H atoms
            feat.append(atom.GetExplicitValence())   # Explicit valence
            feat.append(atom.GetImplicitValence())   # Implicit valence
            feat.append(int(atom.IsInRing()))        # Ring membership
            feat.append(1.5)                         # Default atomic radius
            feat.append(2.0)                         # Default electronegativity
            feat.append(atom.GetMass())              # Atomic mass

            # Pad to 75 dimensions
            while len(feat) < 75:
                feat.append(0)

            atom_feats.append(feat)

        # Create graph
        g = dgl.graph([])
        g.add_nodes(len(atom_feats))

        # Add edges
        src_list = []
        dst_list = []
        for bond in mol.GetBonds():
            src = bond.GetBeginAtomIdx()
            dst = bond.GetEndAtomIdx()
            src_list.extend([src, dst])
            dst_list.extend([dst, src])

        # Ensure that the graph has at least one edge
        if len(src_list) == 0 and len(atom_feats) > 0:
            for i in range(len(atom_feats)):
                src_list.append(i)
                dst_list.append(i)

        g.add_edges(src_list, dst_list)

        # Add node features
        g.ndata['h'] = torch.FloatTensor(atom_feats)

        return g

    def _protein_to_idx(self, protein_seq):
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_dict = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}
        aa_dict['X'] = 0  # Unknown amino acid

        idx_list = [aa_dict.get(aa, 0) for aa in protein_seq]

        if len(idx_list) > self.max_protein_len:
            idx_list = idx_list[:self.max_protein_len]
        else:
            idx_list = idx_list + [0] * (self.max_protein_len - len(idx_list))

        return torch.LongTensor(idx_list)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['SMILES']
        protein = row['Protein']
        label = float(row['Y'])

        # Use existing cluster annotations if available;
        # otherwise generate pseudo cluster IDs
        if 'drug_cluster' in self.data.columns and 'target_cluster' in self.data.columns:
            drug_cluster = int(row['drug_cluster'])
            target_cluster = int(row['target_cluster'])
        else:
            drug_cluster = hash(smiles) % 10
            target_cluster = hash(protein) % 10

        graph = self._smiles_to_graph(smiles)
        if graph is None:
            # If the SMILES string is invalid, return a fallback graph
            graph = dgl.graph([])
            graph.add_nodes(1)
            graph.ndata['h'] = torch.zeros((1, 75))

        protein_tensor = self._protein_to_idx(protein)

        return {
            'graph': graph,
            'protein': protein_tensor,
            'label': torch.FloatTensor([label]),
            'drug_cluster': torch.LongTensor([drug_cluster]),
            'target_cluster': torch.LongTensor([target_cluster])
        }


def _first_existing_path(candidates, patterns, description):
    for path in candidates:
        if path and os.path.isfile(path):
            return path

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted(set(matches))
    if matches:
        return matches[0]

    searched = candidates + patterns
    raise FileNotFoundError(
        f"Cannot find {description}. Checked:\n  " + "\n  ".join(searched)
    )


def _resolve_feature_path(feature_dir, split, kind, strategy=None):
    split_dir = os.path.join(feature_dir, split)
    roots = [split_dir, feature_dir]

    if kind == "drug":
        names = [
            f"{split}_smiles_features.npy",
            f"{split}_drug_features.npy",
            f"{split}_smiles.npy",
        ]
        if strategy:
            names.insert(0, f"{split}_smiles_{strategy}.npy")
        patterns = [os.path.join(root, f"{split}_smiles*.npy") for root in roots]
        patterns += [os.path.join(root, f"{split}_drug*.npy") for root in roots]
        description = f"drug features for split '{split}'"
    elif kind == "protein":
        names = [
            f"{split}_protein_features_esm2.npy",
            f"{split}_protein_features.npy",
            f"{split}_protein_esm2.npy",
            f"{split}_protein.npy",
        ]
        if strategy:
            names.insert(0, f"{split}_protein_{strategy}.npy")
        patterns = [os.path.join(root, f"{split}_protein*esm2*.npy") for root in roots]
        patterns += [os.path.join(root, f"{split}_protein*.npy") for root in roots]
        description = f"protein features for split '{split}'"
    elif kind == "label":
        names = [f"{split}_labels.npy", f"{split}_label.npy", f"{split}_Y.npy"]
        if strategy:
            names.insert(0, f"{split}_labels_{strategy}.npy")
        patterns = [os.path.join(root, f"{split}_labels*.npy") for root in roots]
        patterns += [os.path.join(root, f"{split}_label*.npy") for root in roots]
        description = f"labels for split '{split}'"
    else:
        raise ValueError(f"Unsupported feature kind: {kind}")

    candidates = [os.path.join(root, name) for root in roots for name in names]
    return _first_existing_path(candidates, patterns, description)


class PreExtractedClusterDTIDataset(Dataset):
    """Dataset backed by pre-extracted ChemBERTa/ESM2 features."""

    def __init__(self, feature_dir, split, cluster_csv=None, feature_strategy=None):
        self.feature_dir = os.path.abspath(feature_dir)
        self.split = split
        self.feature_strategy = feature_strategy

        drug_path = _resolve_feature_path(self.feature_dir, split, "drug", feature_strategy)
        protein_path = _resolve_feature_path(self.feature_dir, split, "protein", feature_strategy)
        labels_path = _resolve_feature_path(self.feature_dir, split, "label", feature_strategy)

        self.drug_features = np.load(drug_path, allow_pickle=True)
        self.protein_features = np.load(protein_path, allow_pickle=True)
        self.labels = np.load(labels_path, allow_pickle=True).astype(np.float32)

        n = len(self.labels)
        if len(self.drug_features) != n:
            raise ValueError(f"Drug feature rows ({len(self.drug_features)}) != labels ({n})")
        if len(self.protein_features) != n:
            raise ValueError(f"Protein feature rows ({len(self.protein_features)}) != labels ({n})")

        self.cluster_data = None
        if cluster_csv and os.path.isfile(cluster_csv):
            self.cluster_data = pd.read_csv(cluster_csv)
            if len(self.cluster_data) != n:
                raise ValueError(
                    f"Cluster CSV rows ({len(self.cluster_data)}) != labels ({n}): {cluster_csv}"
                )

        self.drug_dim = self._feature_dim(self.drug_features[0])
        self.protein_dim = self._feature_dim(self.protein_features[0])

        print(
            f"Loaded {split}: samples={n}, drug_dim={self.drug_dim}, "
            f"protein_dim={self.protein_dim}"
        )
        print(f"  drug features: {drug_path}")
        print(f"  protein features: {protein_path}")
        print(f"  labels: {labels_path}")
        if cluster_csv:
            print(f"  clusters: {cluster_csv if self.cluster_data is not None else 'not found; using pseudo clusters'}")

    @staticmethod
    def _feature_vector(feature):
        arr = np.asarray(feature, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("Feature item must be at least 1-D")
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)

    @classmethod
    def _feature_dim(cls, feature):
        return int(cls._feature_vector(feature).shape[0])

    def __len__(self):
        return len(self.labels)

    def _clusters_for_row(self, idx):
        if self.cluster_data is not None:
            row = self.cluster_data.iloc[idx]
            if 'drug_cluster' in self.cluster_data.columns and 'target_cluster' in self.cluster_data.columns:
                return int(row['drug_cluster']), int(row['target_cluster'])

        return idx % 10, (idx * 9973) % 10

    def __getitem__(self, idx):
        drug_feat = torch.from_numpy(self._feature_vector(self.drug_features[idx]))
        protein_feat = torch.from_numpy(self._feature_vector(self.protein_features[idx]))
        label = torch.FloatTensor([float(self.labels[idx])])
        drug_cluster, target_cluster = self._clusters_for_row(idx)

        return {
            'drug_feat': drug_feat,
            'protein_feat': protein_feat,
            'label': label,
            'drug_cluster': torch.LongTensor([drug_cluster]),
            'target_cluster': torch.LongTensor([target_cluster])
        }


def collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    proteins = torch.stack([item['protein'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    drug_clusters = torch.cat([item['drug_cluster'] for item in batch])
    target_clusters = torch.cat([item['target_cluster'] for item in batch])

    batched_graph = dgl.batch(graphs)

    return {
        'graph': batched_graph,
        'protein': proteins,
        'label': labels,
        'drug_cluster': drug_clusters,
        'target_cluster': target_clusters
    }


def pretrained_collate_fn(batch):
    drug_feats = torch.stack([item['drug_feat'] for item in batch])
    protein_feats = torch.stack([item['protein_feat'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    drug_clusters = torch.cat([item['drug_cluster'] for item in batch])
    target_clusters = torch.cat([item['target_cluster'] for item in batch])

    return {
        'drug_feat': drug_feats,
        'protein_feat': protein_feats,
        'label': labels,
        'drug_cluster': drug_clusters,
        'target_cluster': target_clusters
    }


def forward_feature_batch(model, batch, device):
    drug_feats = batch['drug_feat'].to(device)
    protein_feats = batch['protein_feat'].to(device)
    _, _, features, outputs = model(drug_feats, protein_feats, mode="train")
    return features, outputs


class ClusterAwareLoss(nn.Module):
    def __init__(self, lambda_cluster=0.0, lambda_domain=0.0, use_cluster_loss=False):
        super().__init__()
        self.lambda_cluster = lambda_cluster
        self.lambda_domain = lambda_domain
        self.use_cluster_loss = use_cluster_loss
        self.bce = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

    def forward(self, pred, target, drug_cluster=None, target_cluster=None, domain_pred=None, domain_label=None):
        if pred.dim() == 2 and target.dim() == 1:
            target = target.unsqueeze(1)

        task_loss = self.bce(pred, target)

        cluster_loss = 0
        if self.use_cluster_loss and drug_cluster is not None and target_cluster is not None:
            unique_drug_clusters = torch.unique(drug_cluster)
            unique_target_clusters = torch.unique(target_cluster)

            for dc in unique_drug_clusters:
                dc_mask = (drug_cluster == dc)
                if dc_mask.sum() > 1:
                    dc_preds = pred[dc_mask]
                    dc_mean = dc_preds.mean()
                    cluster_loss += ((dc_preds - dc_mean) ** 2).mean()

            for tc in unique_target_clusters:
                tc_mask = (target_cluster == tc)
                if tc_mask.sum() > 1:
                    tc_preds = pred[tc_mask]
                    tc_mean = tc_preds.mean()
                    cluster_loss += ((tc_preds - tc_mean) ** 2).mean()

        domain_loss = 0
        if domain_pred is not None and domain_label is not None:
            domain_label = domain_label.long().view(-1)
            domain_loss = self.domain_criterion(domain_pred, domain_label)

        return task_loss + self.lambda_cluster * cluster_loss + self.lambda_domain * domain_loss


def train_epoch(model, dataloader, optimizer, criterion, device, discriminator=None, optimizer_d=None, epoch=0,
                da_init_epoch=0, random_layer=None, use_entropy=True):
    model.train()
    if discriminator is not None:
        discriminator.train()

    total_loss = 0
    total_domain_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        optimizer.zero_grad()
        if optimizer_d is not None:
            optimizer_d.zero_grad()

        features, outputs = forward_feature_batch(model, batch, device)
        labels = batch['label'].to(device)
        drug_clusters = batch['drug_cluster'].to(device)
        target_clusters = batch['target_cluster'].to(device)

        if discriminator is not None and epoch >= da_init_epoch:
            p = float(epoch - da_init_epoch) / (100 - da_init_epoch)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            domain_label = torch.zeros(features.size(0), dtype=torch.long, device=device)
            if 'is_target' in batch:
                domain_label[batch['is_target'].to(device).bool()] = 1

            task_loss = criterion(outputs, labels, drug_clusters, target_clusters)
            domain_loss = cdan_domain_loss(
                discriminator,
                features,
                outputs,
                domain_label,
                alpha=alpha,
                random_layer=random_layer,
                use_entropy=use_entropy
            )
            loss = task_loss + criterion.lambda_domain * domain_loss
            total_domain_loss += domain_loss.item()
        else:
            loss = criterion(outputs, labels, drug_clusters, target_clusters)

        loss.backward()
        optimizer.step()
        if optimizer_d is not None:
            optimizer_d.step()

        total_loss += loss.item()

        pred_numpy = torch.sigmoid(outputs).detach().cpu().numpy()
        if pred_numpy.ndim == 2:
            pred_numpy = pred_numpy.squeeze(1)
        all_preds.extend(pred_numpy)

        label_numpy = labels.cpu().numpy()
        if label_numpy.ndim == 2:
            label_numpy = label_numpy.squeeze(1)
        all_labels.extend(label_numpy)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

    metrics = {
        'loss': total_loss / len(dataloader),
        'domain_loss': total_domain_loss / len(dataloader) if discriminator is not None else 0,
        'auc': auc,
        'acc': acc
    }

    return metrics


def train_da_epoch(model, source_loader, target_loader, optimizer, criterion, device, discriminator, optimizer_d,
                   epoch, da_init_epoch, lambda_domain=0.1, random_layer=None, use_entropy=True):
    """Train one epoch under cross-domain adaptation."""
    model.train()
    discriminator.train()

    total_loss = 0
    total_domain_loss = 0
    all_source_preds = []
    all_source_labels = []

    n_batches = max(len(source_loader), len(target_loader))
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for _ in range(n_batches):
        try:
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_batch = next(source_iter)

        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)

        optimizer.zero_grad()
        optimizer_d.zero_grad()

        source_labels = source_batch['label'].to(device)
        source_drug_clusters = source_batch['drug_cluster'].to(device)
        source_target_clusters = source_batch['target_cluster'].to(device)

        features_source, outputs_source = forward_feature_batch(model, source_batch, device)

        features_target, outputs_target = forward_feature_batch(model, target_batch, device)

        p = float(epoch - da_init_epoch) / (100 - da_init_epoch)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        task_loss = criterion(outputs_source, source_labels, source_drug_clusters, source_target_clusters)

        if epoch >= da_init_epoch:
            domain_features = torch.cat([features_source, features_target], dim=0)
            domain_outputs = torch.cat([outputs_source, outputs_target], dim=0)
            domain_labels = torch.cat(
                [
                    torch.zeros(features_source.size(0), dtype=torch.long, device=device),
                    torch.ones(features_target.size(0), dtype=torch.long, device=device)
                ],
                dim=0
            )
            domain_loss = cdan_domain_loss(
                discriminator,
                domain_features,
                domain_outputs,
                domain_labels,
                alpha=alpha,
                random_layer=random_layer,
                use_entropy=use_entropy
            )

            loss = task_loss + lambda_domain * domain_loss
            total_domain_loss += domain_loss.item()
        else:
            loss = task_loss

        loss.backward()
        optimizer.step()
        if epoch >= da_init_epoch:
            optimizer_d.step()

        total_loss += loss.item()

        pred_numpy = torch.sigmoid(outputs_source).detach().cpu().numpy()
        if pred_numpy.ndim == 2:
            pred_numpy = pred_numpy.squeeze(1)
        all_source_preds.extend(pred_numpy)

        label_numpy = source_labels.cpu().numpy()
        if label_numpy.ndim == 2:
            label_numpy = label_numpy.squeeze(1)
        all_source_labels.extend(label_numpy)

    all_source_preds = np.array(all_source_preds)
    all_source_labels = np.array(all_source_labels)

    auc = roc_auc_score(all_source_labels, all_source_preds)
    acc = accuracy_score(all_source_labels, (all_source_preds > 0.5).astype(int))

    metrics = {
        'loss': total_loss / n_batches,
        'domain_loss': total_domain_loss / n_batches if epoch >= da_init_epoch else 0,
        'auc': auc,
        'acc': acc
    }

    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)
            drug_clusters = batch['drug_cluster'].to(device)
            target_clusters = batch['target_cluster'].to(device)

            _, outputs = forward_feature_batch(model, batch, device)

            if outputs.dim() == 2 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels.squeeze(), drug_clusters, target_clusters)
            total_loss += loss.item()

            pred_numpy = torch.sigmoid(outputs).cpu().numpy()
            if pred_numpy.ndim == 2:
                pred_numpy = pred_numpy.squeeze(1)
            all_preds.extend(pred_numpy)

            label_numpy = labels.cpu().numpy()
            if label_numpy.ndim == 2:
                label_numpy = label_numpy.squeeze(1)
            all_labels.extend(label_numpy)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'loss': total_loss / len(dataloader),
        'auc': roc_auc_score(all_labels, all_preds),
        'auprc': average_precision_score(all_labels, all_preds),
        'acc': accuracy_score(all_labels, (all_preds > 0.5).astype(int)),
        'precision': precision_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0),
        'recall': recall_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0)
    }
    return metrics


def analyze_cluster_performance(model, dataloader, device):
    model.eval()

    drug_cluster_results = {}
    target_cluster_results = {}

    with torch.no_grad():
        for batch in dataloader:
            labels = np.atleast_1d(batch['label'].cpu().numpy().squeeze())
            drug_clusters = np.atleast_1d(batch['drug_cluster'].cpu().numpy().squeeze())
            target_clusters = np.atleast_1d(batch['target_cluster'].cpu().numpy().squeeze())

            _, outputs = forward_feature_batch(model, batch, device)
            predictions = np.atleast_1d(torch.sigmoid(outputs).cpu().numpy().squeeze())

            for i in range(len(labels)):
                dc = int(drug_clusters[i])
                if dc not in drug_cluster_results:
                    drug_cluster_results[dc] = {'preds': [], 'labels': []}
                drug_cluster_results[dc]['preds'].append(float(predictions[i]))
                drug_cluster_results[dc]['labels'].append(float(labels[i]))

            for i in range(len(labels)):
                tc = int(target_clusters[i])
                if tc not in target_cluster_results:
                    target_cluster_results[tc] = {'preds': [], 'labels': []}
                target_cluster_results[tc]['preds'].append(float(predictions[i]))
                target_cluster_results[tc]['labels'].append(float(labels[i]))

    print("\n===== Drug Cluster Performance Analysis =====")
    drug_cluster_metrics = {}
    for cluster_id, data in drug_cluster_results.items():
        if len(data['labels']) < 5:
            continue
        try:
            labels_array = np.array(data['labels'])
            preds_array = np.array(data['preds'])
            auc = roc_auc_score(labels_array, preds_array)
            acc = accuracy_score(labels_array, (preds_array > 0.5).astype(int))
            drug_cluster_metrics[cluster_id] = {'auc': auc, 'acc': acc, 'count': len(data['labels'])}
            print(f"Drug cluster {cluster_id}: samples={len(data['labels'])}, AUC={auc:.4f}, ACC={acc:.4f}")
        except Exception as e:
            print(f"Drug cluster {cluster_id}: samples={len(data['labels'])}, failed to compute metrics: {str(e)}")

    print("\n===== Target Cluster Performance Analysis =====")
    target_cluster_metrics = {}
    for cluster_id, data in target_cluster_results.items():
        if len(data['labels']) < 5:
            continue
        try:
            labels_array = np.array(data['labels'])
            preds_array = np.array(data['preds'])
            auc = roc_auc_score(labels_array, preds_array)
            acc = accuracy_score(labels_array, (preds_array > 0.5).astype(int))
            target_cluster_metrics[cluster_id] = {'auc': auc, 'acc': acc, 'count': len(data['labels'])}
            print(f"Target cluster {cluster_id}: samples={len(data['labels'])}, AUC={auc:.4f}, ACC={acc:.4f}")
        except Exception as e:
            print(f"Target cluster {cluster_id}: samples={len(data['labels'])}, failed to compute metrics: {str(e)}")

    return drug_cluster_metrics, target_cluster_metrics


def load_data(mode="default", warn_missing=False):
    if mode == "cross_domain":
        # DrugBank-style cross-domain data
        source_train_path = '../datasets/BindingDB/cluster/source_train'
        target_train_path = '../datasets/BindingDB/cluster/target_train'
        target_test_path = '../datasets/BindingDB/cluster/target_test'

        # Check whether files exist
        if warn_missing and not (
            os.path.exists(source_train_path) and
            os.path.exists(target_train_path) and
            os.path.exists(target_test_path)
        ):
            print("The cross-domain data files do not exist. Please ensure the following files are available:")
            print(source_train_path)
            print(target_train_path)
            print(target_test_path)
            return load_data(mode="default")

        return source_train_path, target_train_path, target_test_path
    else:
        # Default data
        train_path = '../datasets/BindingDB/cluster/train_with_clusters'
        val_path = '../datasets/BindingDB/cluster/val_with_clusters'
        test_path = '../datasets/BindingDB/cluster/test_with_clusters'

        # Check whether files exist
        if warn_missing and not (
            os.path.exists(train_path) and
            os.path.exists(val_path) and
            os.path.exists(test_path)
        ):
            print("The default data files do not exist. Please ensure the following files are available:")
            print(train_path)
            print(val_path)
            print(test_path)

        return train_path, val_path, test_path


def main():
    args = parser.parse_args()

    config = {
        "DRUG": {
            "NODE_IN_FEATS": 75,
            "NODE_IN_EMBEDDING": 128,
            "HIDDEN_LAYERS": [128, 128, 128],
            "PADDING": True
        },
        "PROTEIN": {
            "EMBEDDING_DIM": 128,
            "NUM_FILTERS": [128, 128, 128],
            "NUM_HEAD": 4,
            "PADDING": True
        },
        "DECODER": {
            "IN_DIM": 384,
            "HIDDEN_DIM": 768,
            "OUT_DIM": 384,
            "BINARY": 1,
            "DROPOUT_RATE": 0.02
        },
        "PRETRAINED": {
            "USE_ESM2": True,
            "USE_CHEMBERT": True,
            "ESM2_DIM": 1280,
            "CHEMBERT_DIM": 384,
            "FEATURE_DIR": args.feature_dir
        },
        "AGICA_CROSS_ATTENTION": {
            "NUM_HEAD": 4,
            "EMBEDDING_DIM": 128,
            "AGICA_DROPOUT_RATE": 0.1
        },
        "DA": {
            "USE": args.mode == "cross_domain",
            "METHOD": "CDAN",
            "INIT_EPOCH": 10,
            "LAMBDA_DOMAIN": args.lambda_domain,
            "LAMBDA_CLUSTER": args.lambda_cluster,
            "RANDOM_LAYER": True,
            "RANDOM_DIM": 256,
            "ORIGINAL_RANDOM": False,
            "USE_ENTROPY": True
        },
        "SOLVER": {
            "MAX_EPOCH": 100,
            "BATCH_SIZE": 64,
            "LR": 1e-4
        },
        "RESULT": {
            "OUTPUT_DIR": "./results",
            "SAVE_MODEL": True
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running in {args.mode} mode")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    if args.mode == "cross_domain":
        default_source_csv, default_target_train_csv, default_target_test_csv = load_data(mode="cross_domain")

        source_dataset = PreExtractedClusterDTIDataset(
            args.source_feature_dir or args.feature_dir,
            args.source_split,
            cluster_csv=args.source_cluster_csv or default_source_csv,
            feature_strategy=args.feature_strategy
        )
        target_train_dataset = PreExtractedClusterDTIDataset(
            args.target_train_feature_dir or args.feature_dir,
            args.target_train_split,
            cluster_csv=args.target_train_cluster_csv or default_target_train_csv,
            feature_strategy=args.feature_strategy
        )
        target_test_dataset = PreExtractedClusterDTIDataset(
            args.target_test_feature_dir or args.feature_dir,
            args.target_test_split,
            cluster_csv=args.target_test_cluster_csv or default_target_test_csv,
            feature_strategy=args.feature_strategy
        )

        print(f"Source train dataset size: {len(source_dataset)}")
        print(f"Target train dataset size: {len(target_train_dataset)}")
        print(f"Target test dataset size: {len(target_test_dataset)}")

        train_loader = DataLoader(
            source_dataset,
            batch_size=config["SOLVER"]["BATCH_SIZE"],
            shuffle=True,
            collate_fn=pretrained_collate_fn
        )
        valid_loader = DataLoader(
            target_train_dataset,
            batch_size=config["SOLVER"]["BATCH_SIZE"],
            shuffle=True,
            collate_fn=pretrained_collate_fn
        )
        test_loader = DataLoader(
            target_test_dataset,
            batch_size=config["SOLVER"]["BATCH_SIZE"],
            shuffle=False,
            collate_fn=pretrained_collate_fn
        )

        source_loader = train_loader
        target_train_loader = valid_loader
    else:
        default_train_csv, default_val_csv, default_test_csv = load_data()

        train_dataset = PreExtractedClusterDTIDataset(
            args.feature_dir,
            args.train_split,
            cluster_csv=args.train_cluster_csv or default_train_csv,
            feature_strategy=args.feature_strategy
        )
        valid_dataset = PreExtractedClusterDTIDataset(
            args.feature_dir,
            args.val_split,
            cluster_csv=args.val_cluster_csv or default_val_csv,
            feature_strategy=args.feature_strategy
        )
        test_dataset = PreExtractedClusterDTIDataset(
            args.feature_dir,
            args.test_split,
            cluster_csv=args.test_cluster_csv or default_test_csv,
            feature_strategy=args.feature_strategy
        )

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(valid_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["SOLVER"]["BATCH_SIZE"],
            shuffle=True,
            collate_fn=pretrained_collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config["SOLVER"]["BATCH_SIZE"],
            shuffle=False,
            collate_fn=pretrained_collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["SOLVER"]["BATCH_SIZE"],
            shuffle=False,
            collate_fn=pretrained_collate_fn
        )

    datasets_for_dim_check = [train_loader.dataset, valid_loader.dataset, test_loader.dataset]
    drug_dims = {dataset.drug_dim for dataset in datasets_for_dim_check}
    protein_dims = {dataset.protein_dim for dataset in datasets_for_dim_check}
    if len(drug_dims) != 1 or len(protein_dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions: drug={drug_dims}, protein={protein_dims}")

    config["PRETRAINED"]["CHEMBERT_DIM"] = train_loader.dataset.drug_dim
    config["PRETRAINED"]["ESM2_DIM"] = train_loader.dataset.protein_dim
    print(
        f"Using pre-extracted features: drug_dim={config['PRETRAINED']['CHEMBERT_DIM']}, "
        f"protein_dim={config['PRETRAINED']['ESM2_DIM']}"
    )

    model = MAGC_DTI(**config).to(device)

    if config["DA"]["USE"]:
        discriminator_input_dim = (
            config["DA"]["RANDOM_DIM"]
            if config["DA"]["RANDOM_LAYER"]
            else config["DECODER"]["IN_DIM"]
        )
        discriminator = Discriminator(input_size=discriminator_input_dim).to(device)

        if config["DA"]["RANDOM_LAYER"]:
            if config["DA"]["ORIGINAL_RANDOM"]:
                random_layer = RandomLayer(
                    [config["DECODER"]["IN_DIM"], 2],
                    config["DA"]["RANDOM_DIM"]
                ).to(device)
            else:
                random_layer = nn.Linear(
                    config["DECODER"]["IN_DIM"] + 2,
                    config["DA"]["RANDOM_DIM"],
                    bias=False
                ).to(device)
                torch.nn.init.normal_(random_layer.weight, mean=0, std=1)
                for param in random_layer.parameters():
                    param.requires_grad = False
        else:
            random_layer = None

        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["SOLVER"]["LR"])
    else:
        discriminator = None
        random_layer = None
        optimizer_d = None

    optimizer = torch.optim.Adam(model.parameters(), lr=config["SOLVER"]["LR"])

    criterion = ClusterAwareLoss(
        lambda_cluster=config["DA"]["LAMBDA_CLUSTER"],
        lambda_domain=config["DA"]["LAMBDA_DOMAIN"],
        use_cluster_loss=args.use_cluster_loss
    )

    # Training loop
    best_auc = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    model_save_path = 'best_model.pth'

    for epoch in range(config["SOLVER"]["MAX_EPOCH"]):
        print(f"\nEpoch {epoch + 1}/{config['SOLVER']['MAX_EPOCH']}")

        if args.mode == "cross_domain" and config["DA"]["USE"]:
            train_metrics = train_da_epoch(
                model, source_loader, target_train_loader, optimizer, criterion, device,
                discriminator, optimizer_d, epoch, config["DA"]["INIT_EPOCH"],
                lambda_domain=config["DA"]["LAMBDA_DOMAIN"], random_layer=random_layer,
                use_entropy=config["DA"]["USE_ENTROPY"]
            )
            print(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
            if epoch >= config["DA"]["INIT_EPOCH"]:
                print(f"Domain Loss: {train_metrics['domain_loss']:.4f}")
        else:
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device,
                discriminator=discriminator, optimizer_d=optimizer_d,
                epoch=epoch, da_init_epoch=config["DA"]["INIT_EPOCH"],
                random_layer=random_layer, use_entropy=config["DA"]["USE_ENTROPY"]
            )
            print(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")

        val_metrics = evaluate(model, valid_loader, criterion, device)
        print(
            f"Validation - Loss: {val_metrics['loss']:.4f}, "
            f"AUC: {val_metrics['auc']:.4f}, "
            f"AUPRC: {val_metrics['auprc']:.4f}"
        )

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch + 1
            model_save_path = 'best_da_model.pth' if args.mode == "cross_domain" else 'best_model.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved! AUC: {best_auc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. No improvement for {patience} epochs.")
                break

    model.load_state_dict(torch.load(model_save_path))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Test results (from epoch {best_epoch}):")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"AUPRC: {test_metrics['auprc']:.4f}")
    print(f"Accuracy: {test_metrics['acc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"Loss: {test_metrics['loss']:.4f}")

    # Cluster-level performance analysis
    if args.analyze_clusters:
        analyze_cluster_performance(model, test_loader, device)


if __name__ == '__main__':
    main()
