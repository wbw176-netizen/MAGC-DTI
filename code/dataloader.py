import os
from functools import partial

import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    smiles_to_bigraph,
)

from utils import integer_label_protein


class DTIDataset(data.Dataset):

    def __init__(self, list_IDs, df, max_drug_nodes=300):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        row = int(self.list_IDs[index])
        smiles = self.df.iloc[row]["SMILES"]
        graph = self.fc(
            smiles=smiles,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )

        actual_node_feats = graph.ndata.pop("h")
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = max(0, self.max_drug_nodes - num_actual_nodes)

        virtual_node_bit = torch.zeros(
            num_actual_nodes,
            1,
            dtype=actual_node_feats.dtype,
        )
        graph.ndata["h"] = torch.cat(
            (actual_node_feats, virtual_node_bit),
            dim=1,
        )

        if num_virtual_nodes:
            virtual_node_feat = torch.cat(
                (
                    torch.zeros(
                        num_virtual_nodes,
                        actual_node_feats.shape[1],
                        dtype=actual_node_feats.dtype,
                    ),
                    torch.ones(
                        num_virtual_nodes,
                        1,
                        dtype=actual_node_feats.dtype,
                    ),
                ),
                dim=1,
            )
            graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})

        protein = integer_label_protein(self.df.iloc[row]["Protein"])
        label = self.df.iloc[row]["Y"]
        return graph, protein, label


class PreExtractedNpyDTIDataset(data.Dataset):

    def __init__(
        self,
        feature_dir,
        split,
        use_esm2=True,
        use_chembert=True,
        indices=None,
    ):
        if (
            not split
            or os.path.basename(split) != split
            or split in (".", "..")
        ):
            raise ValueError(
                f"split must be a single directory name, got: {split}"
            )

        self.split = split
        self.feature_dir = os.path.abspath(feature_dir)
        self.use_esm2 = use_esm2
        self.use_chembert = use_chembert
        self._h5_file = None

        split_dir = os.path.join(self.feature_dir, split)
        self.h5_path = os.path.join(split_dir, f"{split}_features.h5")
        self.storage_format = (
            "hdf5" if os.path.isfile(self.h5_path) else "npy"
        )

        if self.storage_format == "hdf5":
            self._load_hdf5_metadata()
        else:
            self._load_npy_metadata(split_dir)

        n_samples = len(self._labels_full)
        if indices is None:
            self.indices = np.arange(n_samples, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
            if len(self.indices) and (
                self.indices.min() < 0 or self.indices.max() >= n_samples
            ):
                raise ValueError(
                    f"indices must be within [0, {n_samples - 1}]"
                )

    def _load_hdf5_metadata(self):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required for token-level HDF5 features"
            ) from exc

        with h5py.File(self.h5_path, "r") as h5_file:
            required = {
                "drug_values",
                "drug_offsets",
                "protein_values",
                "protein_offsets",
                "labels",
            }
            missing = required.difference(h5_file.keys())
            if missing:
                raise ValueError(
                    f"Invalid HDF5 feature file; missing {sorted(missing)}"
                )
            self._labels_full = h5_file["labels"][:].astype(np.float32)
            self.drug_dim = int(h5_file["drug_values"].shape[1])
            self.protein_dim = int(h5_file["protein_values"].shape[1])

        print(
            f"Loaded token features: {self.h5_path}, "
            f"samples={len(self._labels_full)}, "
            f"drug_dim={self.drug_dim}, protein_dim={self.protein_dim}"
        )

    def _load_npy_metadata(self, split_dir):
        labels_path = os.path.join(
            split_dir,
            f"{self.split}_labels.npy",
        )
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(
                f"No HDF5 or NPY feature set found in: {split_dir}"
            )
        self._labels_full = np.load(
            labels_path,
            mmap_mode="r",
        ).astype(np.float32)

        if self.use_chembert:
            drug_path = os.path.join(
                split_dir,
                f"{self.split}_smiles_features.npy",
            )
            if not os.path.isfile(drug_path):
                raise FileNotFoundError(
                    f"ChemBERTa features not found: {drug_path}"
                )
            self.drug_features = np.load(drug_path, mmap_mode="r")
            self.drug_dim = int(self.drug_features.shape[-1])
            print(
                f"Loaded ChemBERTa features: {drug_path}, "
                f"shape={self.drug_features.shape}"
            )

        if self.use_esm2:
            protein_path = os.path.join(
                split_dir,
                f"{self.split}_protein_features_esm2.npy",
            )
            if not os.path.isfile(protein_path):
                raise FileNotFoundError(
                    f"ESM-2 features not found: {protein_path}"
                )
            self.protein_features = np.load(
                protein_path,
                mmap_mode="r",
            )
            self.protein_dim = int(self.protein_features.shape[-1])
            print(
                f"Loaded ESM-2 features: {protein_path}, "
                f"shape={self.protein_features.shape}"
            )

        n_samples = len(self._labels_full)
        if self.use_chembert and len(self.drug_features) != n_samples:
            raise ValueError(
                f"Drug feature rows {len(self.drug_features)} "
                f"!= labels {n_samples}"
            )
        if self.use_esm2 and len(self.protein_features) != n_samples:
            raise ValueError(
                f"Protein feature rows {len(self.protein_features)} "
                f"!= labels {n_samples}"
            )

    def _get_h5_file(self):
        if self._h5_file is None:
            import h5py

            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    @staticmethod
    def _read_h5_sequence(h5_file, prefix, row):
        offsets = h5_file[f"{prefix}_offsets"]
        start = int(offsets[row])
        end = int(offsets[row + 1])
        values = h5_file[f"{prefix}_values"][start:end]
        return torch.from_numpy(
            np.asarray(values, dtype=np.float32)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])

        if self.storage_format == "hdf5":
            h5_file = self._get_h5_file()
            drug_feat = self._read_h5_sequence(
                h5_file,
                "drug",
                row,
            )
            protein_feat = self._read_h5_sequence(
                h5_file,
                "protein",
                row,
            )
        else:
            if self.use_chembert:
                drug_feat = torch.from_numpy(
                    np.array(
                        self.drug_features[row],
                        dtype=np.float32,
                        copy=True,
                    )
                )
            else:
                drug_feat = torch.zeros(384, dtype=torch.float32)

            if self.use_esm2:
                protein_feat = torch.from_numpy(
                    np.array(
                        self.protein_features[row],
                        dtype=np.float32,
                        copy=True,
                    )
                )
            else:
                protein_feat = torch.zeros(1280, dtype=torch.float32)

        label = float(self._labels_full[row])
        return drug_feat, protein_feat, label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def close(self):
        h5_file = getattr(self, "_h5_file", None)
        if h5_file is not None:
            h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()


class PreExtractedDTIDataset(PreExtractedNpyDTIDataset):

    def __init__(
        self,
        list_IDs,
        df,
        feature_dir,
        split,
        max_drug_nodes=300,
        use_esm2=True,
        use_chembert=True,
    ):
        del max_drug_nodes
        self.df = df
        super().__init__(
            feature_dir=feature_dir,
            split=split,
            use_esm2=use_esm2,
            use_chembert=use_chembert,
            indices=list_IDs,
        )


class MultiDataLoader:
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = max(1, int(n_batches))
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(loader) for loader in self._dataloaders]

    def _get_nexts(self):
        batches = []
        for index, iterator in enumerate(self._iterators):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self._dataloaders[index])
                self._iterators[index] = iterator
                batch = next(iterator)
            batches.append(batch)
        return batches

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches


def pretrained_collate_func(batch):

    drug_feats, protein_feats, labels = zip(*batch)

    if drug_feats[0].dim() == 1:
        drug_feats = torch.stack(drug_feats)
    else:
        drug_feats = pad_sequence(
            drug_feats,
            batch_first=True,
            padding_value=0.0,
        )

    if protein_feats[0].dim() == 1:
        protein_feats = torch.stack(protein_feats)
    else:
        protein_feats = pad_sequence(
            protein_feats,
            batch_first=True,
            padding_value=0.0,
        )

    labels = torch.tensor(labels, dtype=torch.float32)
    return drug_feats, protein_feats, labels
