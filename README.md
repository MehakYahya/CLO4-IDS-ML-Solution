# Network Intrusion Detection — ML Solution

## Project Title
Network Intrusion Detection — Machine Learning Solution

## Objective
Build, train, evaluate, and save machine learning models to detect network intrusions (malicious traffic) using two datasets included in this repository. The notebooks demonstrate preprocessing, training a Random Forest classifier, evaluating performance, and saving predictions.

## Repository Structure
- `dataset 1/assignment.ipynb` — Notebook using a Kaggle-like Network Intrusion Detection dataset (Train_data.csv / Test_data.csv). Preprocessing, Random Forest training, model save (`rf_model.joblib`).

- `dataset 2/dataset2.ipynb` — Notebook using the UNSW-NB15 dataset (training and testing CSVs). Preprocessing, Random Forest training, evaluation, and confusion matrix plotting.
- `dataset 1/cleaned_dataset.csv` — Cleaned training data produced by the first notebook (if run).


## Dataset Setup Instructions
1. Dataset 1 (Kaggle-like):
-https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection
   - Place the archive (e.g., `archive (6).zip`) in `dataset 1/` or update the path in `dataset 1/assignment.ipynb` at the unzip cell.
   - Expected extracted files: `dataset/Train_data.csv` and `dataset/Test_data.csv` within `dataset 1/`.
   - The notebook `dataset 1/assignment.ipynb` reads `dataset/Train_data.csv` and `dataset/Test_data.csv`. If you already have `cleaned_dataset.csv` and `rf_model.joblib`, the notebook will load them directly.

2. Dataset 2 (UNSW-NB15):
   - https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FUNSW%2DNB15%20dataset%2FCSV%20Files&viewid=f8d1dec5%2Dcd5f%2D42ae%2D8b06%2D2fece580c74a&ga=1
   - Place the UNSW-NB15 CSVs in `dataset 2/UNSW-NB15/` named exactly:
     - `UNSW_NB15_training-set.csv`
     - `UNSW_NB15_testing-set.csv`
   - The notebook `dataset 2/dataset2.ipynb` expects these filenames and paths. Adjust the paths in the notebook if your files are elsewhere.

## Required Packages
A recommended minimal Python environment (tested on Python 3.8+):

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

You can install them with pip. Example (PowerShell):

```powershell
python -m pip install --upgrade pip; \
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

(Optionally create a virtual environment first.)

## How to Run the Code
Open the repository in VS Code or Jupyter and follow these steps.

1. Dataset 1 notebook
   - Open `dataset 1/assignment.ipynb`.
   - Run the cells sequentially. The notebook will:
     - Extract the archive if present.
     - Load `Train_data.csv`, preprocess, normalize, and save `cleaned_dataset.csv`.
     - Train a Random Forest model and save it to `rf_model.joblib`.
     - Load `Test_data.csv`, preprocess, run predictions, and save `predicted_test_data.csv`.

2. Dataset 2 notebook
   - Open `dataset 2/dataset2.ipynb`.
   - Run the cells sequentially. The notebook will:
     - Load UNSW training and testing CSVs, preprocess (impute, encode, scale), train a Random Forest classifier, and evaluate on the test set.

Notes:
- If a required CSV path is wrong, edit the `train_path` / `test_path` variables at the top of the notebooks to point to your local files.
- For reproducibility, both notebooks set random seeds (via model constructors) where applicable.

## Brief Summary of Results
- Dataset 1 (`assignment.ipynb`): Trained a Random Forest classifier (100 estimators). The notebook prints validation accuracy and a classification report, then saves `rf_model.joblib`. Predictions on the test set are saved to `predicted_test_data.csv`.

- Dataset 2 (`dataset2.ipynb`): Trained a Random Forest classifier (200 estimators, class_weight='balanced') on UNSW-NB15. The notebook prints test accuracy, a classification report, and displays a confusion matrix heatmap.

Expected outcomes: Reasonable accuracy for binary normal/attack detection. For multi-class attack categorization, per-class metrics can vary; consult the printed classification reports in each notebook for details.

## Repository Structure
project/
├── dataset 1/
│ ├── assignment.ipynb
│ ├── cleaned_dataset.csv (generated)
│ └── rf_model.joblib (generated)
├── dataset 2/
│ ├── dataset2.ipynb
│ └── UNSW-NB15/
│ ├── UNSW_NB15_training-set.csv
│ └── UNSW_NB15_testing-set.csv

## Next Steps / Recommendations
- Add a `requirements.txt` or `environment.yml` for exact dependency versions.
- Add a small Python script to run preprocessing and model training non-interactively (for automation/CI).
- Save model evaluation metrics (CSV/JSON) for easier reporting.

## Contact / License
- Author: repository owner
- License: Add an appropriate open-source license if you plan to share this project publicly.
