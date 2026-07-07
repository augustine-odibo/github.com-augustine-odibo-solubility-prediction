# github.com-augustine-odibo-solubility-prediction
solubility-prediction
"""
predict.py
----------
Load the trained model and predict aqueous solubility for a new molecule
given as a SMILES string.

Usage:
    python predict.py "CC(=O)OC1=CC=CC=C1C(=O)O"   # aspirin
"""

import sys
import joblib
import pandas as pd

from featurize import smiles_to_features

MODEL_PATH = "results/solubility_model.pkl"


def predict_solubility(smiles: str) -> float:
    feats = smiles_to_features(smiles)
    if feats is None:
        raise ValueError(f"Could not parse SMILES string: {smiles}")

    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([feats])
    pred = model.predict(X)[0]
    return pred


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <SMILES>")
        print('Example: python predict.py "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin')
        sys.exit(1)

    smiles = sys.argv[1]
    logS = predict_solubility(smiles)
    print(f"SMILES: {smiles}")
    print(f"Predicted log solubility: {logS:.2f} mol/L")
