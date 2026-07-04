# MAGC-DTI

MAGC-DTI is a drug-target interaction prediction project. The model combines drug and protein representations, applies AGICA-based cross-modal interaction, and predicts binary interaction labels.

## Environment Setup

Create the conda environment from `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate MAGC_DTI
```

The environment includes PyTorch, DGL, DGLLife, RDKit, Transformers, scikit-learn, and other required packages.

If CUDA or DGL installation fails, install the PyTorch and DGL versions that match your local CUDA driver.

## Downloads

The full datasets files are hosted externally because they are too large for this repository:
- [Download Datasets (Google Drive)](https://drive.google.com/file/d/11mKoGdp-eeQwcHHGHKkYJ--yirD6IYXF/view?usp=drive_link)
- [Download Pretrained datasets (Google Drive)](https://drive.google.com/drive/folders/1NhbLLpFDeEVLd_KqzBvuzURIqyymwJoM?usp=drive_link)
Download and unzip the dataset files into:

```text
datasets/
```

If the Google Drive folder also contains pre-extracted feature files, place them under a feature root such as:

```text
features/<dataset_name>/
```

The supported PLM checkpoints are:

| Modality | Model | Link |
| :--- | :--- | :--- |
| Drug | ChemBERTa | [DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) |
| Target | ESM-2 | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |

Place local PLM weights in a directory such as:

```text
plm_models/
  chemberta_model/
  esm2_model/
```

## Pre-Extracted Feature Format

Token-level features are stored as HDF5 files:

```text
features_root/
  train/train_features.h5
  val/val_features.h5
  test/test_features.h5
```

Pooled features are stored as `.npy` files:

```text
features_root/
  train/
    train_smiles_features.npy
    train_protein_features_esm2.npy
    train_labels.npy
  val/
  test/
```

Typical dimensions are:

```text
ChemBERTa: 384
ESM2:      1280
```

## Feature Extraction

```bash
cd code

python pre_data_extractor.py \
  --data_dir ../datasets/drugbank/random2 \
  --output_dir ../features/drugbank \
  --representation token \
  --batch_size 4
```

Use `--representation pooled` for pooled `.npy` features.

## Training

Run raw-input training:

```bash
cd code
python main.py --data drugbank --split random2
```

Run training with pre-extracted ChemBERTa and ESM2 features:

```bash
cd code
python main.py \
  --use_pretrained_features \
  --feature_dir ../features/drugbank \
  --run_name drugbank_plm
```

Training outputs are written under:

```text
output/result/<dataset>/<split>/
```

Feature file not found

Check that your feature directory follows the required split structure:

```text
train/train_features.h5
val/val_features.h5
test/test_features.h5
```

Dimension mismatch in pre-extracted feature mode

Make sure train, validation, and test splits use the same ChemBERTa and ESM2 feature dimensions.

## References

1. Chithrananda, S., Grand, G., & Ramsundar, B. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. arXiv:2010.09885.
2. Lin, Z., Akin, H., Rao, R., et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637), 1123-1130.
3. Elnaggar, A., Heinzinger, M., Dallago, C., et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE TPAMI, 44(10), 7112-7127.
4. McInnes, L., Healy, J., & Melville, J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
5. Wishart, D. S., et al. DrugBank 5.0: a major update to the DrugBank database for 2018. Nucleic Acids Research, 46(D1), D1074-D1082.
6. Bai, P., Miljković, F., John, B., & Lu, H. (2023). Interpretable bilinear attention network with domain adaptation improves drug–target prediction. Nature Machine Intelligence. https://doi.org/10.1038/s42256-022-00605-1
