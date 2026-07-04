from model import MAGC_DTI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, PreExtractedNpyDTIDataset, pretrained_collate_func
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime
cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
parser = argparse.ArgumentParser(description="MAGC-DTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='default mode', default='BindingDB')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split name", choices=['random', 'random2', 'random3', 'random1','cold','unseen_drug','unseen_target'])
parser.add_argument('--pre_dir', type=str, default=None, help="pre-extracted features directory name")
parser.add_argument('--amp', action='store_true', help='Activate AMP (Automatic Mixed Precision) training')
parser.add_argument('--output_dir',type=str, metavar='DIR', help='output directory', default='random3')
parser.add_argument('--use_pretrained_features', action='store_true', help='use preextracted features')
parser.add_argument('--feature_dir', type=str, default='../Drugbank/mean_mean', help='pre-extracted features directory path')
parser.add_argument('--run_name', type=str, default=None, help='optional output tag for pre-extracted feature runs')
args = parser.parse_args()


def _run_tag_from_feature_dir(feature_dir):
    p = os.path.normpath(os.path.abspath(feature_dir))
    parts = [x for x in p.replace("\\", "/").split("/") if x]
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1] if parts else "pretrained_npy"


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)

    if args.use_pretrained_features:
        cfg.PRETRAINED.USE_ESM2 = True
        cfg.PRETRAINED.USE_CHEMBERT = True
        cfg.PRETRAINED.FEATURE_DIR = args.feature_dir
        run_tag = args.run_name or _run_tag_from_feature_dir(args.feature_dir)
        data_name, split_name = "pretrained", run_tag
        mkdir(cfg.RESULT.OUTPUT_DIR + f"{data_name}/{split_name}")
        train_dataset = PreExtractedNpyDTIDataset(
            args.feature_dir, "train",
            use_esm2=cfg.PRETRAINED.USE_ESM2,
            use_chembert=cfg.PRETRAINED.USE_CHEMBERT,
        )
        val_dataset = PreExtractedNpyDTIDataset(
            args.feature_dir, "val",
            use_esm2=cfg.PRETRAINED.USE_ESM2,
            use_chembert=cfg.PRETRAINED.USE_CHEMBERT,
        )
        test_dataset = PreExtractedNpyDTIDataset(
            args.feature_dir, "test",
            use_esm2=cfg.PRETRAINED.USE_ESM2,
            use_chembert=cfg.PRETRAINED.USE_CHEMBERT,
        )
        feature_datasets = (train_dataset, val_dataset, test_dataset)
        drug_dims = {dataset.drug_dim for dataset in feature_datasets}
        protein_dims = {
            dataset.protein_dim for dataset in feature_datasets
        }
        if len(drug_dims) != 1 or len(protein_dims) != 1:
            raise ValueError(
                "Feature dimensions differ across train/val/test: "
                f"drug={drug_dims}, protein={protein_dims}"
            )
        cfg.PRETRAINED.CHEMBERT_DIM = train_dataset.drug_dim
        cfg.PRETRAINED.ESM2_DIM = train_dataset.protein_dim
        train_collate_fn = pretrained_collate_func
    else:
        data_name, split_name = args.data, args.split
        mkdir(cfg.RESULT.OUTPUT_DIR + f"{data_name}/{split_name}")
        print("start...")
        print(f"dataset:{args.data}, split CSV:{args.split}")
        dataFolder = os.path.join("../datasets", args.data, args.split)
        train_path = os.path.join(dataFolder, "train.csv")
        val_path = os.path.join(dataFolder, "val.csv")
        test_path = os.path.join(dataFolder, "test.csv")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        train_dataset = DTIDataset(df_train.index.values, df_train)
        val_dataset = DTIDataset(df_val.index.values, df_val)
        test_dataset = DTIDataset(df_test.index.values, df_test)
        train_collate_fn = graph_collate_func

    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    
    print(f'train_dataset:{len(train_dataset)}')
    print(f'val_dataset:{len(val_dataset)}')
    print(f'test_dataset:{len(test_dataset)}')

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': 8,
                                                               'drop_last':True, 'collate_fn': train_collate_fn}

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = MAGC_DTI(device=device, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True

    if args.amp:
        print("Activate AMP (Automatic Mixed Precision) training")
    
    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator,
                      data_name, split_name, use_amp=args.amp,
                      use_pretrained_features=args.use_pretrained_features, **cfg)
    result = trainer.train()

    out_sub = f"{data_name}/{split_name}"
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{out_sub}/model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{out_sub}/config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))

    print(f"\nDirectory for saving result: {cfg.RESULT.OUTPUT_DIR}{out_sub}")
    print(f'\nend...')

    return result


if __name__ == '__main__':
    torch.cuda.empty_cache()
    print(f"start time: {datetime.now()}")
    s = time()
    result = main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s, ")
