# LIGHT: Automated discovery of optimal hydrogel components by the intelligent agent

## Features

### 1. LIGHT_platform Agent for Automated Processes.、

The LIGHT_platform Agent encompasses three core tasks/functionalities for machine learning model training and results generation, covering Unsupervised Learning, Supervised Learning (Regression & Classification), and the entire process of Best Agent Selection.

### 2. Database Construction Module (Automated Data Extraction)

This project builds a high-throughput literature data extraction system using the DeepSeek-32B model and PDF parsing tools (such as pdfplumber).

### 3. Final prediction

Hydrogel candidate components screened automatically by the LIGHT_platform.

### 4. Supervised Learning Module

The supervised module establishes nonlinear mapping relationships between molecular structures and material properties.

Using automatically extracted data, we train regression and classification models to predict Young’s modulus and swelling ratio, respectively.

### 5. Unsupervised Learning Module

This module explores the distribution and clustering patterns of hydrogels in the “structure–property space.”

## Installation

### 1. Clone this repository:
```bash
git clone https://github.com/UFO-Group/LIGHT_platform/
cd LIGHT_platform
```

### 2. Install dependencies

(Different environments for different modules)

#### For literature extraction:

```bash
cd Automated_Data_Extraction
pip install -r requirements.txt
```

#### For Unsupervised Learning:

```bash
cd Unsupervised_Learning
pip install -r requirements.txt
```

#### For Supervised Learning:
```bash
cd Supervised_Learning
pip install -r requirements.txt
```

## Usage

### LIGHT_platform Agent
Automated screening of hydrogel candidate components using a program and an LLM.
 
### Automated\_Data\_Extraction
Use the DeepSeek API to extract information from literature. See README inside Automated\_Data\_Extraction for details.

### Unsupervised\_Learning
Use automatically extracted data to perform unsupervised learning and determine combinations. See README in Unsupervised\_Learning.

### Supervised\_Learning
Use automatically extracted data to train regression and classification models to predict Young’s modulus and swelling ratio. See README in Supervised\_Learning.

## File Structure

```text
LIGHT_platform
│  README.md
│  
├─Agent
│  │  Agent_unsupervised.py							— Automated Unsupervised pipline
│  │  Agent_workflow.py								— Main workflow of LIGHT_platform
│  │  Selection_API.py								— API used for Selection 
│  │  Selection_pipeline.py							— Automated Selection pipeline
│  │  Supervised_pipline.py							— Automated Supervised pipline
│  │  README.md
│  │  __init__.py
│          
├─Automated_Data_Extraction							— Automated data extraction folder
│  │  pipeline_config.json					
│  │  pipeline_main.py					
│  │  README.md					
│  │  requirements.txt					
│  │  Split_pdf.py									— PDF splitting
│  │  Standardize_Units.py							— Standardize units in tables
│  │  Table_Generation.py							— Convert extracted information into tables
│  │  			
│  │      			
│  ├─Data			
│  │  ├─Data_split			
│  │  │        			
│  │  ├─PDFs			
│  │  │      			
│  │  └─Processed_Results			
│  │          			
│  ├─Data_Extraction								— Literature extraction scripts
│  │  │  API_SwellingRatio.py						— API script for Swelling Ratio
│  │  │  API_YoungsModulus.py						— API script for Young's Modulus
│  │  │  Clean.py			
│  │  │  contains_keywords_swelling.py			
│  │  │  contains_keywords_youngs.py			
│  │  │  main_PDF_Swellingratio.py					— main extract script for Swelling Ratio
│  │  │  main_PDF_Youngsmodulus.py					— main extract script for Young's Modulus
│  │  │  TextNormalizer.py
│  │          
│  ├─Download
│     │  Doi_Abstract_Swelling_search.py
│     │  Doi_Abstract_Youngs_search.py
│     │  PDFs_download.py
│             
│          
├─Final_predction
│      Agent_Final_Selection.csv
│      
├─Supervised_Learning								— Supervised learning folder
│  │  requirements.txt			
│  │  			
│  ├─Classification_Model							— Classification model scripts
│  │  │  classification_main.py			
│  │  │  draw_Matrix.py			
│  │  │  draw_pipline.py			
│  │  │  draw_ROC.py			
│  │  │  morgan_pooling.py							— Morgan generate and pooling script
│  │  │  pipeline.py			
│  │  │  predict.py									— Swelling Ratio prediction script
│  │  │  README_classification.txt			
│  │  │  train_rf.py								— main RF-Regression script
│  │  │  
│  │  ├─Best_result
│  │  │      
│  │  └─results
│  │      └─SwellingRatio
│  │          │  SwellingRatio_predict.csv 
│  │          ├─draw
│  │          │                  
│  │          ├─features
│  │          │      
│  │          └─rf_cls_cv10_t9
│  │                      
│  ├─DataBase										— Original database and data distribution scripts
│  │  │  Swelling_Distribution_Statistics_Plot.py
│  │  │  swelling_ratio.csv
│  │  │  Youngs_Distribution_Statistics_Plot.py
│  │  │  youngs_modulus.csv
│  │  
│  │          
│  ├─High-throughput predict						— Unsupervised results and prediction scripts
│  │      kmeans-pooled.csv			
│  │      kmeans_results.csv			
│  │      README.md			
│  │      Result-swelling.csv			
│  │      Result-youngs.csv			
│  │      			
│  └─Regression_Model								— Regression model scripts
│      │  draw_pipline.py
│      │  README.md
│      │  regression_main.py
│      │  run_pipeline.py
│      │  
│      ├─API
│      │  │  advise_best_model_with_api.py
│      │  │  API.py
│      │  
│      │          
│      ├─Best_result
│      │  ├─MLP
│      │  │      
│      │  ├─OLS
│      │  │      
│      │  ├─RF
│      │  │      
│      │  └─SVM
│      │          
│      ├─draw
│      │      draw_r2.py
│      │      
│      ├─grid
│      │      grid_mlp.py							— run a grid search for MLP
│      │      grid_svm.py               			— run a grid search for SVM
│      │      rf_grid_loop.py           			— run a grid search for RF
│      │      
│      ├─main_regression
│      │      baseline_mlp_svm.py					— mlp and svm model
│      │      baseline_OLS_linear_regression.py     — OLS linear regression model
│      │      baseline_RF.py                        — RF model
│      │      morgan_pooling.py                     — Morgan generate and pooling script
│      │      train_mlp_svm_pipeline.py
│      │      
│      ├─predict
│      │      predict.py							— Young's Modulus prediction script
│      │      
│      └─results
│          └─YoungsModulus
│              │  model_candidates_for_llm.json
│              │  overall_best_model.json
│              │  
│              ├─draw
│              │              
│              ├─features
│              │      
│              ├─mlp_grid
│              │      
│              ├─ols_linear
│              │          
│              ├─predictions
│              │      
│              ├─rf_cv10
│              │          
│              ├─rf_grid
│              │      
│              ├─runs
│              │  ├─mlp
│              │  │          
│              │  └─svm
│              │              
│              └─svm_grid
│                      
└─Unsupervised_Learning											— Unsupervised learning folder
    │  morgan_pooling.py										— Morgan generate and pooling script
    │  README.md
    │  requirements.txt
    │  
    ├─candidate_umap											— Candidate component distribution plots
    │      all_AB_smiles2morgan.py								— Generate all SMILES pairs and 1024-bit Morgan fingerprints
    │      all_random_smiles_AB_concat1024.rar
    │      candidate_umap_automation.py							— Automatically achieve the umap distribution of candidate components
    │      candidate_umap_coordinates.csv
    │      cluster-3-AB-morgan.py								— Read the candidate components file and output Morgan fingerprints
    │      Prediction-1028-ALL2-1024.npy
    │      Prediction-1028-ALL2-candidate-1024.npy
    │      Prediction-1028-ALL2-candidate-process.csv
    │      Prediction-1028-ALL2-candidate.csv
    │      Prediction-1028-ALL2-process.csv
    │      Prediction-1028-ALL2.csv
    │      smiles_count.csv
    │      smiles_count_random_2.csv
    │      umap-candidate.py									— Used to generate the UMAP distribution plot for candidate components
    │      umap_candidate_visualization.png
    │      
    ├─database													— Original database
    │      swelling_ratio.csv
    │      youngs_modulus.csv
    │      
    └─unsupervised_learning										— Scripts for unsupervised clustering
        │  AB_concat1024.npy
        │  analyze_unsupervised.py								— Analyze the unsupervised results
        │  API.py												— Invoke the large language model
        │  Best_classification_response.py						— Select the optimal cluster using the large language model
        │  cluster_umap_kmeans.png
        │  cluster_umap_kmeans_from_npy.png
        │  data-process.py										— Used for merging the two databases of Young's modulus and swelling ratio
        │  final_two_smiles_with_modulus.csv
        │  morgan.py											— Generate 1024-length fingerprints from the merged database
        │  smiles_count_AB.py									— Combining and removing duplicates for the aggregation polymer judgments of the best clusters
        │  umap2d-kmeans.py										— Additionally, running the command below directly reproduces the unsupervised classification results in the paper
        │  umap2d.npy
        │  unsupervised.py										— Perform unsupervised clustering
        │  unsupervised_learning_automation.py					— Automated implementation of unsupervised learning and analysis
        │  
        ├─clusters												— Unsupervised classification results
        │      cluster_0.csv
        │      cluster_1.csv
        │      cluster_2.csv
        │      cluster_3-AB-ordered.csv
        │      cluster_3-AB-unique.csv
        │      cluster_3-counts.csv
        │      cluster_3.csv
        │      cluster_4.csv
        │      cluster_5.csv
        │      cluster_statistics.csv
        │      
        └─clusters-from-umap									— Reproduce the unsupervised results of the article
                cluster_0.csv
                cluster_1.csv
                cluster_2.csv
                cluster_3.csv
                cluster_4.csv
                cluster_5.csv
```
## About
Developed by:

UFO Group,

China, Donghua University.

## License
This project is licensed under the MIT License.






