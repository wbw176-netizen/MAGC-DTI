#!/usr/bin/env python3
"""Extract ChemBERTa and ESM-2 features for MAGC-DTI.

Two representations are supported:

* token: variable-length token embeddings stored compactly in HDF5.
* pooled: one mean-pooled vector per sample stored in NPY files.
"""

import argparse
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger("magc_dti_feature_extractor")
SPLITS = ("train", "val", "test")


class PretrainedFeatureExtractor:
    """ChemBERTa + ESM-2 feature extractor."""

    def __init__(
        self,
        chemberta_model,
        esm2_model,
        device="auto",
        use_amp=True,
        include_special_tokens=False,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.include_special_tokens = include_special_tokens
        self.chemberta_name = chemberta_model
        self.esm2_name = esm2_model

        model_kwargs = {}
        if self.use_amp:
            model_kwargs["torch_dtype"] = torch.float16

        LOGGER.info("Loading ChemBERTa from %s", chemberta_model)
        self.drug_tokenizer = AutoTokenizer.from_pretrained(
            chemberta_model
        )
        self.drug_model = AutoModel.from_pretrained(
            chemberta_model,
            **model_kwargs,
        ).to(self.device)
        self.drug_model.eval()

        LOGGER.info("Loading ESM-2 from %s", esm2_model)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(esm2_model)
        self.protein_model = AutoModel.from_pretrained(
            esm2_model,
            **model_kwargs,
        ).to(self.device)
        self.protein_model.eval()

        self.drug_dim = int(self.drug_model.config.hidden_size)
        self.protein_dim = int(self.protein_model.config.hidden_size)
        LOGGER.info(
            "Models ready on %s: drug_dim=%d, protein_dim=%d",
            self.device,
            self.drug_dim,
            self.protein_dim,
        )

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
        return nullcontext()

    def _encode_batch(
        self,
        texts,
        tokenizer,
        model,
        max_length,
        representation,
    ):
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        special_tokens_mask = encoded.pop(
            "special_tokens_mask",
            None,
        )
        attention_mask = encoded["attention_mask"].bool()
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode(), self._autocast_context():
            hidden = model(**encoded).last_hidden_state

        hidden = hidden.float().cpu()
        valid_mask = attention_mask.cpu()
        if (
            not self.include_special_tokens
            and special_tokens_mask is not None
        ):
            valid_mask = valid_mask & ~special_tokens_mask.bool()

        features = []
        for row in range(hidden.shape[0]):
            row_mask = valid_mask[row]
            if not row_mask.any():
                row_mask = attention_mask[row].cpu()
            token_features = hidden[row, row_mask]

            if representation == "pooled":
                features.append(token_features.mean(dim=0).numpy())
            else:
                features.append(token_features.numpy())

        return features

    @staticmethod
    def _append_sequences(dataset, sequences, dtype):
        arrays = [
            np.asarray(sequence, dtype=dtype)
            for sequence in sequences
        ]
        lengths = [len(array) for array in arrays]
        if not arrays:
            return lengths

        values = np.concatenate(arrays, axis=0)
        start = dataset.shape[0]
        dataset.resize(start + values.shape[0], axis=0)
        dataset[start:] = values
        return lengths

    @staticmethod
    def _create_value_dataset(
        h5_file,
        name,
        feature_dim,
        dtype,
        compression,
    ):
        return h5_file.create_dataset(
            name,
            shape=(0, feature_dim),
            maxshape=(None, feature_dim),
            chunks=(min(256, max(1, 65536 // feature_dim)), feature_dim),
            dtype=dtype,
            compression=compression,
        )

    def extract_pooled_split(
        self,
        dataframe,
        split,
        output_dir,
        batch_size,
        smiles_max_length,
        protein_max_length,
        storage_dtype,
    ):
        drug_batches = []
        protein_batches = []

        for start in tqdm(
            range(0, len(dataframe), batch_size),
            desc=f"{split}: pooled PLM features",
        ):
            batch = dataframe.iloc[start:start + batch_size]
            drug_batches.extend(
                self._encode_batch(
                    batch["SMILES"].astype(str).tolist(),
                    self.drug_tokenizer,
                    self.drug_model,
                    smiles_max_length,
                    "pooled",
                )
            )
            protein_batches.extend(
                self._encode_batch(
                    batch["Protein"].astype(str).tolist(),
                    self.protein_tokenizer,
                    self.protein_model,
                    protein_max_length,
                    "pooled",
                )
            )

        drug_features = np.asarray(
            drug_batches,
            dtype=storage_dtype,
        )
        protein_features = np.asarray(
            protein_batches,
            dtype=storage_dtype,
        )
        labels = dataframe["Y"].to_numpy(dtype=np.float32)

        np.save(
            output_dir / f"{split}_smiles_features.npy",
            drug_features,
        )
        np.save(
            output_dir / f"{split}_protein_features_esm2.npy",
            protein_features,
        )
        np.save(output_dir / f"{split}_labels.npy", labels)

        return {
            "samples": len(dataframe),
            "drug_shape": list(drug_features.shape),
            "protein_shape": list(protein_features.shape),
        }

    def extract_token_split(
        self,
        dataframe,
        split,
        output_dir,
        batch_size,
        smiles_max_length,
        protein_max_length,
        storage_dtype,
        compression,
    ):
        output_path = output_dir / f"{split}_features.h5"
        temporary_path = output_path.with_suffix(".h5.tmp")
        if temporary_path.exists():
            temporary_path.unlink()

        drug_offsets = [0]
        protein_offsets = [0]

        try:
            with h5py.File(temporary_path, "w") as h5_file:
                drug_values = self._create_value_dataset(
                    h5_file,
                    "drug_values",
                    self.drug_dim,
                    storage_dtype,
                    compression,
                )
                protein_values = self._create_value_dataset(
                    h5_file,
                    "protein_values",
                    self.protein_dim,
                    storage_dtype,
                    compression,
                )

                for start in tqdm(
                    range(0, len(dataframe), batch_size),
                    desc=f"{split}: token PLM features",
                ):
                    batch = dataframe.iloc[start:start + batch_size]
                    drug_sequences = self._encode_batch(
                        batch["SMILES"].astype(str).tolist(),
                        self.drug_tokenizer,
                        self.drug_model,
                        smiles_max_length,
                        "token",
                    )
                    protein_sequences = self._encode_batch(
                        batch["Protein"].astype(str).tolist(),
                        self.protein_tokenizer,
                        self.protein_model,
                        protein_max_length,
                        "token",
                    )

                    drug_lengths = self._append_sequences(
                        drug_values,
                        drug_sequences,
                        storage_dtype,
                    )
                    protein_lengths = self._append_sequences(
                        protein_values,
                        protein_sequences,
                        storage_dtype,
                    )

                    for length in drug_lengths:
                        drug_offsets.append(drug_offsets[-1] + length)
                    for length in protein_lengths:
                        protein_offsets.append(
                            protein_offsets[-1] + length
                        )

                h5_file.create_dataset(
                    "drug_offsets",
                    data=np.asarray(drug_offsets, dtype=np.int64),
                )
                h5_file.create_dataset(
                    "protein_offsets",
                    data=np.asarray(protein_offsets, dtype=np.int64),
                )
                h5_file.create_dataset(
                    "labels",
                    data=dataframe["Y"].to_numpy(dtype=np.float32),
                )
                h5_file.attrs["representation"] = "token"
                h5_file.attrs["drug_model"] = self.chemberta_name
                h5_file.attrs["protein_model"] = self.esm2_name
                h5_file.attrs["drug_dim"] = self.drug_dim
                h5_file.attrs["protein_dim"] = self.protein_dim
                h5_file.attrs["include_special_tokens"] = (
                    self.include_special_tokens
                )

            os.replace(temporary_path, output_path)
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

        return {
            "samples": len(dataframe),
            "drug_tokens": int(drug_offsets[-1]),
            "protein_tokens": int(protein_offsets[-1]),
            "drug_average_length": float(
                np.mean(np.diff(drug_offsets))
            ),
            "protein_average_length": float(
                np.mean(np.diff(protein_offsets))
            ),
            "file": str(output_path),
        }

    def extract_all_splits(
        self,
        data_dir,
        output_dir,
        representation,
        batch_size,
        smiles_max_length,
        protein_max_length,
        storage_dtype,
        compression,
        splits,
        overwrite,
    ):
        data_dir = Path(data_dir).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "representation": representation,
            "chemberta_model": self.chemberta_name,
            "esm2_model": self.esm2_name,
            "drug_dim": self.drug_dim,
            "protein_dim": self.protein_dim,
            "storage_dtype": np.dtype(storage_dtype).name,
            "smiles_max_length": smiles_max_length,
            "protein_max_length": protein_max_length,
            "include_special_tokens": self.include_special_tokens,
            "splits": {},
        }

        for split in splits:
            csv_path = data_dir / f"{split}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            dataframe = pd.read_csv(csv_path)
            required_columns = {"SMILES", "Protein", "Y"}
            missing = required_columns.difference(dataframe.columns)
            if missing:
                raise ValueError(
                    f"{csv_path} is missing columns: {sorted(missing)}"
                )

            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            expected_output = (
                split_dir / f"{split}_features.h5"
                if representation == "token"
                else split_dir / f"{split}_smiles_features.npy"
            )
            if expected_output.exists() and not overwrite:
                raise FileExistsError(
                    f"Output exists: {expected_output}; use --overwrite"
                )

            if representation == "token":
                split_metadata = self.extract_token_split(
                    dataframe,
                    split,
                    split_dir,
                    batch_size,
                    smiles_max_length,
                    protein_max_length,
                    storage_dtype,
                    compression,
                )
            else:
                split_metadata = self.extract_pooled_split(
                    dataframe,
                    split,
                    split_dir,
                    batch_size,
                    smiles_max_length,
                    protein_max_length,
                    storage_dtype,
                )
            metadata["splits"][split] = split_metadata

        metadata_path = output_dir / "feature_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        LOGGER.info("Feature metadata written to %s", metadata_path)
        return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ChemBERTa and ESM-2 features for MAGC-DTI."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing train.csv, val.csv and test.csv.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Feature output root.",
    )
    parser.add_argument(
        "--representation",
        choices=("token", "pooled"),
        default="token",
        help="token preserves sequence embeddings; pooled stores one vector.",
    )
    parser.add_argument(
        "--chemberta_model",
        default="DeepChem/ChemBERTa-77M-MLM",
        help="Hugging Face model id or local ChemBERTa directory.",
    )
    parser.add_argument(
        "--esm2_model",
        default="facebook/esm2_t33_650M_UR50D",
        help="Hugging Face model id or local ESM-2 directory.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--smiles_max_length", type=int, default=512)
    parser.add_argument("--protein_max_length", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "lzf", "gzip"),
        default="lzf",
        help="HDF5 compression for token features.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda or a device such as cuda:1.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    parser.add_argument(
        "--include_special_tokens",
        action="store_true",
        help="Keep CLS/EOS tokens in token or pooled representations.",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA mixed precision during extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing feature files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    storage_dtype = (
        np.float16 if args.dtype == "float16" else np.float32
    )
    compression = (
        None if args.compression == "none" else args.compression
    )

    extractor = PretrainedFeatureExtractor(
        chemberta_model=args.chemberta_model,
        esm2_model=args.esm2_model,
        device=args.device,
        use_amp=not args.no_amp,
        include_special_tokens=args.include_special_tokens,
    )
    extractor.extract_all_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        representation=args.representation,
        batch_size=args.batch_size,
        smiles_max_length=args.smiles_max_length,
        protein_max_length=args.protein_max_length,
        storage_dtype=storage_dtype,
        compression=compression,
        splits=args.splits,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
