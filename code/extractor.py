#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import EsmTokenizer, EsmModel
import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
import itertools
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiPoolingFeatureExtractor:
    
    def __init__(self, 
                 protein_model_name: str = "facebook/esm2_t33_650M_UR50D",
                 smiles_model_name: str = "DeepChem/ChemBERTa-77M-MLM"):
        
        self.protein_model_name = protein_model_name
        self.smiles_model_name = smiles_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pooling_strategies = {
            "cls": "Use [CLS] token only",
            "mean": "Mean pooling",
            "max": "Max pooling",
            "cls_mean": "[CLS] + mean pooling",
            "cls_max": "[CLS] + max pooling",
            "mean_max": "Mean pooling + max pooling",
            "all": "Combination of all pooling strategies"
        }
        
        logger.info("Initializing multi-pooling feature extractor")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Supported pooling strategies: {list(self.pooling_strategies.keys())}")
    
    def _load_models(self):
        """Load pretrained models."""
        logger.info("Loading ESM2 model...")
        self.protein_tokenizer = EsmTokenizer.from_pretrained(self.protein_model_name)
        self.protein_model = EsmModel.from_pretrained(self.protein_model_name)
        self.protein_model.to(self.device)
        self.protein_model.eval()
        self.protein_dim = self.protein_model.config.hidden_size
        
        logger.info("Loading ChemBERTa model...")
        self.smiles_tokenizer = AutoTokenizer.from_pretrained(self.smiles_model_name)
        self.smiles_model = AutoModel.from_pretrained(self.smiles_model_name)
        self.smiles_model.to(self.device)
        self.smiles_model.eval()
        self.smiles_dim = self.smiles_model.config.hidden_size
        
        logger.info(f"ESM2 feature dimension: {self.protein_dim}")
        logger.info(f"ChemBERTa feature dimension: {self.smiles_dim}")
    
    def _pool_features(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, 
                      strategy: str) -> torch.Tensor:
        """Implement multiple pooling strategies."""
        
        if strategy == "cls":
            # Use [CLS] token only
            return hidden_states[:, 0, :]
        
        elif strategy == "mean":
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif strategy == "max":
            # Max pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9
            return torch.max(hidden_states, 1)[0]
        
        elif strategy == "cls_mean":
            # [CLS] + mean pooling
            cls_features = hidden_states[:, 0, :]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_features = sum_embeddings / sum_mask
            return (cls_features + mean_features) / 2
        
        elif strategy == "cls_max":
            # [CLS] + max pooling
            cls_features = hidden_states[:, 0, :]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9
            max_features = torch.max(hidden_states, 1)[0]
            return (cls_features + max_features) / 2
        
        elif strategy == "mean_max":
            # Mean pooling + max pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_features = sum_embeddings / sum_mask
            
            hidden_states[input_mask_expanded == 0] = -1e9
            max_features = torch.max(hidden_states, 1)[0]
            return (mean_features + max_features) / 2
        
        elif strategy == "all":
            # Combine all pooled features
            cls_features = hidden_states[:, 0, :]
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_features = sum_embeddings / sum_mask
            
            hidden_states[input_mask_expanded == 0] = -1e9
            max_features = torch.max(hidden_states, 1)[0]
            
            # Concatenate all pooled representations
            return torch.cat([cls_features, mean_features, max_features], dim=1)
        
        else:
            raise ValueError(f"Unsupported pooling strategy: {strategy}")
    
    def extract_protein_features(self, protein_sequences: List[str], 
                               pooling_strategy: str, batch_size: int = 8) -> np.ndarray:
        """Extract protein features."""
        logger.info(f"Extracting protein features - strategy: {pooling_strategy}")
        
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(protein_sequences), batch_size), 
                         desc=f"Protein Features-{pooling_strategy}"):
                batch_sequences = protein_sequences[i:i + batch_size]
                
                # Tokenization
                batch_encoded = self.protein_tokenizer(
                    batch_sequences,
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                batch_encoded = {k: v.to(self.device) for k, v in batch_encoded.items()}
                
                # Forward pass
                outputs = self.protein_model(**batch_encoded)
                
                # Pool features
                batch_features = self._pool_features(
                    outputs.last_hidden_state, 
                    batch_encoded['attention_mask'],
                    pooling_strategy
                ).cpu().numpy()
                
                all_features.append(batch_features)
        
        features = np.vstack(all_features)
        logger.info(f"Protein feature extraction completed - shape: {features.shape}")
        return features
    
    def extract_smiles_features(self, smiles_list: List[str], 
                              pooling_strategy: str, batch_size: int = 16) -> np.ndarray:
        """Extract SMILES features."""
        logger.info(f"Extracting SMILES features - strategy: {pooling_strategy}")
        
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), 
                         desc=f"SMILES Features-{pooling_strategy}"):
                batch_smiles = smiles_list[i:i + batch_size]
                
                # Tokenization
                batch_encoded = self.smiles_tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                batch_encoded = {k: v.to(self.device) for k, v in batch_encoded.items()}
                
                # Forward pass
                outputs = self.smiles_model(**batch_encoded)
                
                # Pool features
                batch_features = self._pool_features(
                    outputs.last_hidden_state, 
                    batch_encoded['attention_mask'],
                    pooling_strategy
                ).cpu().numpy()
                
                all_features.append(batch_features)
        
        features = np.vstack(all_features)
        logger.info(f"SMILES feature extraction completed - shape: {features.shape}")
        return features
    
    def extract_features_with_strategy(self, csv_path: str, 
                                     protein_pooling: str, smiles_pooling: str,
                                     output_dir: str, batch_size: int = 8) -> Dict[str, np.ndarray]:
        """Extract features using the specified pooling strategy."""
        
        # Load input data
        df = pd.read_csv(csv_path)
        smiles_list = df['SMILES'].tolist()
        protein_list = df['Protein'].tolist()
        labels = df['Y'].values if 'Y' in df.columns else None
        
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Number of SMILES: {len(smiles_list)}")
        logger.info(f"Number of proteins: {len(protein_list)}")
        
        # Extract features
        smiles_features = self.extract_smiles_features(smiles_list, smiles_pooling, batch_size)
        protein_features = self.extract_protein_features(protein_list, protein_pooling, batch_size)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse file name
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        strategy_name = f"{protein_pooling}_{smiles_pooling}"
        
        # Organize extracted features
        features_dict = {
            'smiles_features': smiles_features,
            'protein_features': protein_features,
            'labels': labels,
            'smiles_list': smiles_list,
            'protein_list': protein_list,
            'strategy': strategy_name
        }
        
        # Save features to disk
        np.save(os.path.join(output_dir, f"{file_name}_smiles_{strategy_name}.npy"), smiles_features)
        np.save(os.path.join(output_dir, f"{file_name}_protein_{strategy_name}.npy"), protein_features)
        
        if labels is not None:
            np.save(os.path.join(output_dir, f"{file_name}_labels_{strategy_name}.npy"), labels)
        
        # Save metadata
        metadata = {
            'file_name': file_name,
            'strategy': strategy_name,
            'protein_pooling': protein_pooling,
            'smiles_pooling': smiles_pooling,
            'total_samples': len(smiles_list),
            'smiles_feature_dim': smiles_features.shape[1],
            'protein_feature_dim': protein_features.shape[1],
            'model_info': {
                'smiles_model': self.smiles_model_name,
                'protein_model': self.protein_model_name
            },
            'extraction_time': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{file_name}_metadata_{strategy_name}.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature saving completed - strategy: {strategy_name}")
        logger.info(f"SMILES feature shape: {smiles_features.shape}")
        logger.info(f"Protein feature shape: {protein_features.shape}")
        
        return features_dict
    
    def batch_extract_all_strategies(self, csv_path: str, 
                                   output_base_dir: str = "features_pooling_comparison",
                                   batch_size: int = 8,
                                   strategies_to_test: Optional[List[str]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """Batch extract features for all pooling strategy combinations."""
        
        if strategies_to_test is None:
            strategies_to_test = ["cls", "mean", "max", "cls_mean"]
        
        logger.info(f"Starting batch feature extraction - strategies: {strategies_to_test}")
        logger.info(f"CSV file: {csv_path}")
        logger.info(f"Output directory: {output_base_dir}")
        
        # Load models
        self._load_models()
        
        # Generate all strategy combinations
        strategy_combinations = list(itertools.product(strategies_to_test, strategies_to_test))
        logger.info(f"Total number of strategy combinations: {len(strategy_combinations)}")
        
        all_results = {}
        
        for i, (protein_pooling, smiles_pooling) in enumerate(strategy_combinations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing combination {i+1}/{len(strategy_combinations)}: {protein_pooling} + {smiles_pooling}")
            logger.info(f"{'='*60}")
            
            try:
                # Create strategy-specific output directory
                strategy_output_dir = os.path.join(output_base_dir, f"{protein_pooling}_{smiles_pooling}")
                
                # Extract features
                start_time = time.time()
                features = self.extract_features_with_strategy(
                    csv_path=csv_path,
                    protein_pooling=protein_pooling,
                    smiles_pooling=smiles_pooling,
                    output_dir=strategy_output_dir,
                    batch_size=batch_size
                )
                
                extraction_time = time.time() - start_time
                logger.info(f"✓ Completed - time elapsed: {extraction_time:.2f} seconds")
                
                all_results[f"{protein_pooling}_{smiles_pooling}"] = features
                
            except Exception as e:
                logger.error(f"✗ Failed: {e}")
                continue
        
        # Save summary report
        self._save_comparison_report(all_results, output_base_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info("Batch feature extraction completed")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully completed: {len(all_results)} strategy combinations")
        logger.info(f"Output directory: {output_base_dir}")
        
        return all_results
    
    def _save_comparison_report(self, all_results: Dict[str, Dict[str, np.ndarray]], 
                              output_base_dir: str):
        """Save the comparison report."""
        
        report = {
            'extraction_summary': {
                'total_strategies': len(all_results),
                'extraction_time': pd.Timestamp.now().isoformat()
            },
            'strategy_results': {}
        }
        
        for strategy_name, features in all_results.items():
            report['strategy_results'][strategy_name] = {
                'smiles_feature_shape': features['smiles_features'].shape,
                'protein_feature_shape': features['protein_features'].shape,
                'total_samples': features['smiles_features'].shape[0],
                'smiles_feature_dim': features['smiles_features'].shape[1],
                'protein_feature_dim': features['protein_features'].shape[1]
            }
        
        # Save report to disk
        report_path = os.path.join(output_base_dir, "pooling_strategy_comparison_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to: {report_path}")

def main():
    """Main function for batch testing all pooling strategies."""
    
    # Initialize extractor
    extractor = MultiPoolingFeatureExtractor()
    
    # Path to test CSV file
    test_csv = "../datasets/Drugbank/random/train.csv"
    
    if not os.path.exists(test_csv):
        logger.error(f"Test file does not exist: {test_csv}")
        return
    
    # Batch extract all strategy combinations
    results = extractor.batch_extract_all_strategies(
        csv_path=test_csv,
        output_base_dir="../result/features/pooling_strategy_comparison",
        batch_size=4,
        strategies_to_test=["cls", "mean", "max", "cls_mean"]
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Feature extraction summary")
    logger.info("="*80)
    
    for strategy_name, features in results.items():
        logger.info(f"\nStrategy: {strategy_name}")
        logger.info(f"  SMILES features: {features['smiles_features'].shape}")
        logger.info(f"  Protein features: {features['protein_features'].shape}")
        logger.info(f"  Number of samples: {features['smiles_features'].shape[0]}")

if __name__ == "__main__":
    main()
