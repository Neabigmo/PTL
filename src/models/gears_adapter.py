"""
GEARS Adapter for Transportability Benchmark

This module provides a wrapper around GEARS that:
1. Loads data from our processed split artifacts
2. Converts to GEARS-compatible format
3. Runs training and prediction
4. Produces output compatible with our evaluation pipeline
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add sw_mgli site-packages to path for GEARS imports
SW_MGLI_SITE = r"E:\anaconda3\envs\sw_mgli\Lib\site-packages"
if SW_MGLI_SITE not in sys.path:
    sys.path.insert(0, SW_MGLI_SITE)

try:
    from gears import GEARS, PertData
    from gears.utils import print_sys
    GEARS_AVAILABLE = True
except ImportError as e:
    print(f"GEARS import failed: {e}")
    GEARS_AVAILABLE = False

# Constants
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TRAINING_LOG_DIR = RESULTS_DIR / "logs" / "training"


def load_pertdata_from_split(split_json_path: str, data_path: str) -> dict:
    """
    Load our split data and convert to GEARS PertData format.

    Parameters
    ----------
    split_json_path : str
        Path to our split JSON file
    data_path : str
        Path to processed data

    Returns
    -------
    dict with keys: adata, train_conditions, val_conditions, test_conditions
    """
    with open(split_json_path, 'r') as f:
        split_info = json.load(f)

    # Load the h5ad data
    h5ad_path = Path(data_path)
    if not h5ad_path.exists():
        # Try to find it
        h5ad_path = ROOT / "data" / "processed" / split_info.get('dataset', '') / "perturb_processed.h5ad"

    try:
        import scanpy as sc
        adata = sc.read_h5ad(str(h5ad_path))
    except Exception as e:
        raise ValueError(f"Failed to load h5ad from {h5ad_path}: {e}")

    # Extract condition splits
    train_conditions = split_info.get('train_conditions', [])
    val_conditions = split_info.get('val_conditions', [])
    test_conditions = split_info.get('test_conditions', [])

    return {
        'adata': adata,
        'train_conditions': train_conditions,
        'val_conditions': val_conditions,
        'test_conditions': test_conditions,
        'split_info': split_info
    }


def create_gears_pertdata_from_anndata(adata, split_info: dict,
                                         data_path: str) -> PertData:
    """
    Create a GEARS PertData object from our processed AnnData.

    Parameters
    ----------
    adata : AnnData
        The processed AnnData object
    split_info : dict
        Split information from JSON
    data_path : str
        Path to save GEARS data

    Returns
    -------
    PertData object ready for GEARS training
    """
    pertdata = PertData(data_path=data_path)

    # Set the adata
    pertdata.adata = adata
    pertdata.dataset_name = split_info.get('dataset', 'custom')
    pertdata.dataset_path = data_path

    # Set perturbation genes
    pertdata.set_pert_genes()

    return pertdata


class GearsAdapter:
    """
    Adapter for running GEARS on our benchmark data.

    This class handles:
    - Data conversion from our format to GEARS format
    - Model training with our split structure
    - Prediction and output generation
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.pertdata = None

    def setup_from_split(self, split_json_path: str, data_path: str) -> None:
        """
        Set up GEARS from our split JSON.

        Parameters
        ----------
        split_json_path : str
            Path to split JSON file
        data_path : str
            Path to data directory
        """
        # Load our split data
        split_data = load_pertdata_from_split(split_json_path, data_path)
        adata = split_data['adata']

        # Create a temporary directory for GEARS
        temp_dir = Path(RESULTS_DIR) / "temp" / f"gears_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save adata to temp location
        adata_path = temp_dir / "perturb_processed.h5ad"
        adata.write_h5ad(str(adata_path))

        # Create PertData
        self.pertdata = create_gears_pertdata_from_anndata(
            adata, split_data['split_info'], str(temp_dir)
        )

        # Initialize GEARS model
        self.gears = GEARS(self.pertdata, device=self.device)

        # Initialize model with default parameters
        self.gears.model_initialize()

    def train(self, epochs: int = 20, lr: float = 1e-3,
              weight_decay: float = 5e-4) -> dict[str, Any]:
        """
        Train the GEARS model.

        Parameters
        ----------
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        weight_decay : float
            Weight decay

        Returns
        -------
        dict with training metrics
        """
        if self.gears is None:
            raise RuntimeError("Model not initialized. Call setup_from_split first.")

        self.gears.train(epochs=epochs, lr=lr, weight_decay=weight_decay)

        return {"status": "trained", "device": self.device}

    def predict(self, pert_list: list[list[str]]) -> dict[str, np.ndarray]:
        """
        Make predictions for given perturbations.

        Parameters
        ----------
        pert_list : list of perturbation tuples
            e.g., [['gene1'], ['gene2'], ['gene1', 'gene2']]

        Returns
        -------
        dict mapping perturbation string to predicted expression profile
        """
        if self.gears is None:
            raise RuntimeError("Model not initialized.")

        results = self.gears.predict(pert_list)
        return results

    def predict_test(self) -> pd.DataFrame:
        """
        Predict on test set conditions.

        Returns
        -------
        DataFrame with predictions indexed by condition
        """
        if self.gears is None:
            raise RuntimeError("Model not initialized.")

        # Get test conditions from pertdata
        test_conditions = self.pertdata.dataloader.get('test_conditions', [])

        # Make predictions
        pert_list = [[c] for c in test_conditions]
        results = self.predict(pert_list)

        # Convert to DataFrame
        df = pd.DataFrame(results).T
        df.columns = self.pertdata.gene_list

        return df


def run_gears_on_split(split_json_path: str, output_dir: str,
                      epochs: int = 20,
                      device: str = None) -> dict[str, Any]:
    """
    Run full GEARS pipeline on a single split.

    Parameters
    ----------
    split_json_path : str
        Path to split JSON file
    output_dir : str
        Directory to save outputs
    epochs : int
        Number of training epochs
    device : str
        Device to use ('cuda' or 'cpu')

    Returns
    -------
    dict with run status and metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "training.log"

    try:
        # Initialize adapter
        adapter = GearsAdapter(device=device)

        # Load and setup
        print_sys(f"Loading split from {split_json_path}")
        adapter.setup_from_split(split_json_path, str(ROOT / "data" / "processed"))

        # Train
        print_sys(f"Training GEARS for {epochs} epochs")
        adapter.train(epochs=epochs)

        # Predict on test
        print_sys("Making predictions on test set")
        predictions = adapter.predict_test()

        # Save predictions
        pred_path = output_path / "test_predictions.npz"
        np.savez(pred_path, predictions=predictions.values,
                 condition_names=predictions.index.values,
                 gene_names=predictions.columns.values)

        # Save metadata
        metadata = {
            "run_timestamp": datetime.now().isoformat(),
            "device": device,
            "epochs": epochs,
            "n_test_conditions": len(predictions),
            "n_genes": len(predictions.columns)
        }
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            "status": "success",
            "predictions_path": str(pred_path),
            "metadata": metadata
        }

    except Exception as e:
        # Log error
        error_msg = f"GEARS run failed: {str(e)}"
        print_sys(error_msg)

        with open(log_file, 'w') as f:
            f.write(f"ERROR: {error_msg}\n")

        return {
            "status": "failed",
            "error": error_msg
        }


# For command-line usage
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run GEARS on benchmark splits")
    parser.add_argument("--split-json", required=True,
                        help="Path to split JSON file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--device", default=None,
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    result = run_gears_on_split(
        args.split_json,
        args.output_dir,
        args.epochs,
        args.device
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()