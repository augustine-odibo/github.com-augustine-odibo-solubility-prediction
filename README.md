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


    improvement:
    # Solubility Prediction

A machine learning project for predicting molecular solubility using [your approach: neural networks/ensemble methods/etc.].

## Overview

This project predicts the solubility of chemical compounds based on molecular descriptors/SMILES strings [adjust based on your actual input].

## Features

- [Key feature 1]
- [Key feature 2]
- [Key feature 3]

## Installation

```bash
git clone https://github.com/augustine-odibo/github.com-augustine-odibo-solubility-prediction.git
cd github.com-augustine-odibo-solubility-prediction
pip install -r requirements.txt
```

## Usage

```python
from model import predict_solubility

# Example usage
result = predict_solubility(molecule_data)
print(f"Predicted solubility: {result}")
```

## Dataset

- Source: [Your dataset source]
- Size: [Number of samples]
- Features: [Brief description of features]

## Model Performance

- Accuracy: [Your metric]
- RMSE: [Your metric]
- R² Score: [Your metric]

## Project Structure

```
├── data/              # Dataset files
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
└── requirements.txt   # Dependencies
```

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- [Other dependencies]

## License

[Specify your license]

## Author

Augustine Odibo
