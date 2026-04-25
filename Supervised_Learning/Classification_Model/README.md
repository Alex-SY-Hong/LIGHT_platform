## Classification Model Workflow

This folder provides the supervised-learning workflow used for binary property classification. The workflow converts polymer/formulation SMILES into pooled Morgan-fingerprint features, converts a continuous property into a binary class label using a user-defined threshold, trains a Random Forest classification model, evaluates model performance by 10-fold cross-validation, selects the best fold model, predicts new candidate formulations, and generates confusion-matrix and ROC plots.

The current example is based on `Swelling Ratio (times)` classification. In the default setting, samples with swelling ratio greater than or equal to the threshold are assigned to class 1, while samples below the threshold are assigned to class 0.

---

## 1. Folder structure

```text
LIGHT_platform-main/
└── Supervised_Learning/
    ├── DataBase/
    │   └── swelling_ratio.csv        # Raw classification database.
    │
    └── Classification_Model/
        ├── classification_main.py     # One-click classification pipeline: Morgan + training + best-fold selection + prediction + plotting.
        ├── pipeline.py                # Builds binary labels from a continuous property and calls train_rf.py.
        ├── train_rf.py                # Random Forest training script for classification.
        ├── morgan_pooling.py          # Converts SMILES/formulation columns into pooled Morgan fingerprint features.
        ├── predict.py                 # Applies best_model.joblib to new candidates and merges predictions with the source CSV.
        ├── draw_pipline.py            # Automatically plots the best-fold confusion matrix and ROC curves.
        ├── draw_Matrix.py             # Draws confusion-matrix figures.
        ├── draw_ROC.py                # Draws ROC curves.
        ├── README_classification.txt
        │
        └── results/
            └── SwellingRatio/
                ├── features/          # Morgan feature CSV and classification CSV.
                ├── rf_cls_cv10_t9/    # RF 10-fold classification outputs for threshold = 9.
                ├── draw/              # Confusion Matrix and ROC figures.
                └── SwellingRatio_predict.csv
```

---

## 2. Data flow and file format

The classification workflow contains four major data stages:

1. The original SMILES/formulation CSV is converted into Morgan fingerprint features.
2. The continuous target column, such as `Swelling Ratio (times)`, is converted into a binary label column.
3. The generated classification feature CSV is used for RF classification training.
4. The selected best model is used to predict new candidates and export a CSV-format classification database.

---

### 2.1 Input file for Morgan fingerprint generation

The input file for `morgan_pooling.py` should be a CSV table containing SMILES columns, optional ID columns, and a continuous property column.

For the current swelling-ratio classification task, the source property column is:

```text
Swelling Ratio (times)
```

A minimal input table is:

| SMILE A | SMILE B | SMILE C | Swelling Ratio (times) |
|---|---|---|---:|
| `[*]NC(CCCCNC(C(C)=C)=O)C(N1CCCC1C(NCC(N2CC(O)CC2C(NC(COC(C(C)=C)=O)C(NCC([*])=O)=O)=O)=O)=O)=O` | `[*]OCC[*]` | `OCC(C(C(C1N)O)O[*])OC1OC2C(OC(C(C2O)N)[*])CO` | 5.699 |
| `C/C(C[*])=C/C[*]` |  |  | 3.049 |
| ... | ... | ... | ... |

The SMILES strings are converted into a 1024-dimensional pooled Morgan fingerprint vector. The ID-related columns and the continuous source property column are preserved in the Morgan feature CSV when detected or specified.

The Morgan feature CSV has the following structure:

| row_index | Swelling Ratio (times) | fp_0 | fp_1 | ... | fp_1023 |
|---:|---:|---:|---:|---|---:|
| 0 | 5.699 | 0 | 1 | ... | 0 |
| 1 | 3.049 | 2 | 0 | ... | 1 |
| ... | ... | ... | ... | ... | ... |

Here, `fp_0` to `fp_1023` are the Morgan fingerprint features.

---

### 2.2 Binary classification label generation

The script `pipeline.py` converts the continuous source column into a binary class label.

For example, with:

```text
src_col   = Swelling Ratio (times)
threshold = 9
class_col = y_class
```

the label is defined as:

| Condition | Class label |
|---|---:|
| `Swelling Ratio (times) >= 9` | 1 |
| `Swelling Ratio (times) < 9` | 0 |

After label generation, the original continuous column is removed from the classification feature CSV to avoid information leakage.

The generated classification CSV has the following structure:

| row_index | fp_0 | fp_1 | ... | fp_1023 | y_class | _fp_hash |
|---:|---:|---:|---|---:|---:|---|
| 0 | 0 | 1 | ... | 0 | 0 | `hash_value` |
| 1 | 2 | 0 | ... | 1 | 1 | `hash_value` |
| ... | ... | ... | ... | ... | ... | ... |

The `_fp_hash` column is generated from numerical fingerprint features and can be used for group-aware splitting when duplicated fingerprints exist.

---

### 2.3 Input file for RF classification training

The generated classification CSV is used as the input file for `train_rf.py`.

The model-training script automatically uses numerical feature columns as input variables and uses the specified class label column as the classification target.

The main training outputs include:

| Output file / folder | Description |
|---|---|
| `metrics.json` | Single-split training and validation metrics. |
| `feature_importance.csv` | Feature importance from the RF model. |
| `confusion_matrix.csv` | Validation confusion matrix for single-split training. |
| `cv10_metrics.csv` | Per-fold validation metrics for 10-fold CV. |
| `cv10_summary.json` | Mean and standard deviation of CV metrics. |
| `cv10_oof.csv` | Out-of-fold predictions. |
| `cv10/fold_XX_model.joblib` | RF model saved for each CV fold. |
| `cv10/fold_XX_train.csv` | Training predictions for fold XX. |
| `cv10/fold_XX_valid.csv` | Validation predictions for fold XX. |
| `cv10/fold_XX_metrics.json` | Train/validation metrics for fold XX. |
| `cv10/best_model.joblib` | Best fold model selected by `Acc_class_1`. |
| `fold_models/best_model.joblib` | A copied best model for downstream use. |

The key classification metrics include:

```text
Acc
Acc_class_0
Acc_class_1
F1_macro
F1_weighted
BalAcc
Precision
Recall
ROC_AUC
```

By default, `classification_main.py` selects the best fold model using the highest `Acc_class_1`.

---

### 2.4 Prediction output and CSV-format classification database

After model training, the selected best model should be named:

```text
best_model.joblib
```

and placed under the directory specified by `--model_dir`.

For example:

```text
results/SwellingRatio/rf_cls_cv10_t9/cv10/best_model.joblib
```

The prediction output is a CSV-format classification database. A minimal output format is:

The prediction output is a CSV-format classification database. A minimal output format is:

| Pair_ID | SMILE_A | SMILE_B | row_index | Prediction | Prediction_prob_class0 | Prediction_prob_class1 |
|---|---|---|---:|---:|---:|---:|
| Pair_1 | `[*]NC(C(N1CCCC1C(NCC(N2CC(CC2C(NC(C(NCC([*])=O)=O)CO)=O)O)=O)=O)=O)CCCCN` | `[*]NC(CCCCNC(C(C)=C)=O)C(N1CCCC1C(NCC(N2CC(O)CC2C(NC(COC(C(C)=C)=O)C(NCC([*])=O)=O)=O)=O)=O)=O` | 0 | 1 | 0.3957611613965208 | 0.6042388386034796 |
| Pair_2 | `OCC(C(C(C1N)O)O[*])OC1OC2C(OC(C(C2O)N)[*])CO` | `[*]NC(C(N1CCCC1C(NCC(N2CC(CC2C(NC(C(NCC([*])=O)=O)CO)=O)O)=O)=O)=O)CCCCN` | 1 | 1 | 0.3286716238281687 | 0.6713283761718325 |
| ... | ... | ... | ... | ... | ... | ... |

Here:

| Column | Description |
|---|---|
| `Pair_ID` | Identifier of the candidate polymer pair. |
| `SMILE_A` | SMILES string of component A. |
| `SMILE_B` | SMILES string of component B. |
| `row_index` | Row index used to merge the feature table and source table. |
| `Prediction` | Predicted class label. In this workflow, `1` indicates `Swelling Ratio (times) >= threshold`, and `0` indicates `Swelling Ratio (times) < threshold`. |
| `Prediction_prob_class0` | Predicted probability for class 0. |
| `Prediction_prob_class1` | Predicted probability for class 1. |

If `--target_name SwellingRatio_pred` is used, the output columns will be:

```text
SwellingRatio_pred
SwellingRatio_pred_prob_class0
SwellingRatio_pred_prob_class1
```

Additional columns from the original source CSV can also be preserved in the final prediction output.

---

## 3. Top-level scripts

| Script | Role | Main function |
|---|---|---|
| `classification_main.py` | One-click classification pipeline | Runs Morgan feature generation, calls `pipeline.py`, selects the best fold model based on `Acc_class_1`, optionally predicts new candidates, and optionally generates plots. |
| `pipeline.py` | Classification data builder and training controller | Converts continuous `Swelling Ratio (times)` into `y_class`, removes the continuous column to avoid leakage, adds `_fp_hash`, selects a grouping column if possible, and calls `train_rf.py`. |
| `train_rf.py` | RF training script | Trains Random Forest models for classification, supports 10-fold CV, saves fold models, exports predictions, metrics and feature importance. |
| `predict.py` | Prediction script | Loads `best_model.joblib`, predicts classes and class probabilities, and merges them with the original SMILES/formulation table. |
| `draw_pipline.py` | Plotting controller | Finds the fold with the highest `Acc_class_1` and calls `draw_Matrix.py` and `draw_ROC.py`. |
| `draw_Matrix.py` | Confusion-matrix plotting | Generates train and validation confusion-matrix figures. |
| `draw_ROC.py` | ROC plotting | Generates train ROC, validation ROC and train-vs-validation ROC figures. |

Their relationship is:

```text
classification_main.py
│
├── morgan_pooling.py
│
├── pipeline.py
│   └── train_rf.py
│
├── predict.py
│
└── draw_pipline.py
    ├── draw_Matrix.py
    └── draw_ROC.py
```

In practice:

```text
Use classification_main.py when you want to run the complete classification workflow.
Use pipeline.py when Morgan features already exist and you only want to build labels and train RF classification.
Use draw_pipline.py when training results already exist and you only want to regenerate Confusion Matrix and ROC plots.
```

---

## 4. Running examples

All commands below should be executed under:

```text
LIGHT_platform-main/Supervised_Learning/Classification_Model
```

---

### 4.1 One-click full classification pipeline

The recommended one-click entry point is:

```bash
python classification_main.py \
  --task_name SwellingRatio \
  --raw_csv ../DataBase/swelling_ratio.csv \
  --src_col "Swelling Ratio (times)" \
  --threshold 9 \
  --class_col y_class \
  --polymer_cols "SMILE A" "SMILE B" "SMILE C" \
  --do_predict \
  --predict_in_csv "../High-throughput predict/kmeans-pooled.csv" \
  --predict_source_csv "../High-throughput predict/kmeans_results.csv" \
  --predict_out_csv results/SwellingRatio/SwellingRatio_predict.csv
```

This command performs the following steps:

1. Generate 1024-dimensional pooled Morgan fingerprint features.
2. Convert `Swelling Ratio (times)` into the binary label `y_class` using threshold 9.
3. Train an RF classification model with 10-fold CV.
4. Select the fold with the highest `Acc_class_1`.
5. Save the selected model as `best_model.joblib`.
6. Predict new candidates and export `SwellingRatio_predict.csv`.
7. Generate confusion-matrix and ROC figures for the best fold.

---

### 4.2 Generate Morgan fingerprints manually

```bash
python morgan_pooling.py \
  --in_csv ../DataBase/swelling_ratio.csv \
  --polymer_cols "SMILE A" "SMILE B" "SMILE C" \
  --target_col "Swelling Ratio (times)" \
  --alpha 3 \
  --radius 3 \
  --nbits 1024 \
  --out_csv results/SwellingRatio/features/swelling_ratio_morgan.csv
```

---

### 4.3 Run RF classification

```bash
python pipeline.py \
  --in_csv  results/SwellingRatio/features/swelling_ratio_morgan.csv \
  --src_col "Swelling Ratio (times)" \
  --threshold 9 \
  --use_cv10 1
  --base_save_root results/SwellingRatio
```

---

### 4.4 Predict new candidates

Before prediction, make sure that `--model_dir` contains:

```text
best_model.joblib
```

For example:

```text
results/SwellingRatio/rf_cls_cv10_t9/cv10/best_model.joblib
```

Then run:

```bash
python predict.py \
  --in_csv ../High-throughput predict/kmeans-pooled.csv \
  --source_csv ../High-throughput predict/kmeans_results.csv \
  --out_csv results/SwellingRatio/SwellingRatio_predict.csv \
  --model_dir results/SwellingRatio/rf_cls_cv10_t9/cv10/
```

---

### 4.5 Draw Confusion Matrix and ROC automatically

```bash
python draw_pipline.py \
  --task_dir results/SwellingRatio \
  --skip_existing
```

This script automatically finds the fold with the highest `Acc_class_1`, locates the corresponding `fold_XX_train.csv` and `fold_XX_valid.csv`, and generates Confusion Matrix and ROC plots.

---

### 4.6 Draw Confusion Matrix manually

```bash
python draw_Matrix.py \
  --csv_train  results/SwellingRatio/rf_cls_cv10_t9/cv10/fold_06_train.csv \
  --csv_test    results/SwellingRatio/rf_cls_cv10_t9/cv10/fold_06_valid.csv \
  --y_col       y_true \
  --yhat_col    y_pred \
  --out_train   results/SwellingRatio/draw/rf/fold_06/CM/confmat_train.png \
  --out_test    results/SwellingRatio/draw/rf/fold_06/CM/confmat_test.png \
  --out_dir     results/SwellingRatio/draw/rf/fold_06/CM/ \
  --cmap        Blues \
  --rotate_xticks 0 \
  --normalize   none
```

---

### 4.7 Draw ROC manually

```bash
python draw_ROC.py \
  --csv_train results/SwellingRatio/rf_cls_cv10_t9/cv10/fold_06_train.csv \
  --csv_test  results/SwellingRatio/rf_cls_cv10_t9/cv10/fold_06_valid.csv \
  --out_dir   results/SwellingRatio/draw/rf/fold_06/ROC \
  --train_color 109,109,255 \
  --test_color  "#F3A5D9" \
  --fill

```

---

## 5. Expected output structure

After running the one-click workflow, the expected output structure is:

```text
results/
└── SwellingRatio/
    ├── features/
    │   ├── swelling_ratio_morgan.csv
    │   └── swelling_ratio_morgan_CLS9.csv
    │
    ├── rf_cls_cv10_t9/
    │   ├── metrics.json
    │   ├── cv10_metrics.csv
    │   ├── cv10_summary.json
    │   ├── cv10_oof.csv
    │   ├── feature_importance.csv
    │   ├── cv10/
    │   │   ├── fold_01_model.joblib
    │   │   ├── fold_01_train.csv
    │   │   ├── fold_01_valid.csv
    │   │   ├── fold_01_metrics.json
    │   │   ├── ...
    │   │   └── best_model.joblib
    │   └── fold_models/
    │       └── best_model.joblib
    │
    ├── draw/
    │   └── rf/
    │       └── fold_XX/
    │           ├── CM/
    │           │   ├── confmat_train.png
    │           │   └── confmat_valid.png
    │           └── ROC/
    │               ├── roc_train.png
    │               ├── roc_test.png
    │               └── roc_train_vs_test.png
    │
    └── SwellingRatio_predict.csv
```
