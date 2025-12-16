- After downloading the PDFs, run pipeline_main.py to first split each document into smaller sections, then based on your keywords, use the split PDFs to run main_PDF_Youngsmodulus.py for automated data extraction. Finally the Table_Generation.py will generate the output CSV files, and the Standardize_Units.py will standardize the table.

#### The specific process is as follows:

- Preprocessing and Segmentation: The PDF documents undergo a segmentation step, where each file is systematically partitioned into five-page subsets. These segmented files are subsequently staged in the LIGHT_platform-main\Automated_Data_Extraction\Data\Data_split directory.

- Information Extraction: A Large Language Model (LLM) is then employed to perform the automated extraction of pertinent data from the segmented subsets.

- Post-Processing and Standardization: The extracted raw data is finalized through two critical post-processing scripts:

- Table_Generation.py is utilized for the structural organization of the extracted information.

- Standardize_Units.py is implemented to ensure the uniformity and standardization of all reported units.

- Archival of Results: The finalized and standardized results are then stored in the designated output directory: LIGHT_platform-main\Automated_Data_Extraction\Data\Processed_Results.


#### How to run Data_Extraction:

```bash
cd LIGHT_platform-main\Automated_Data_Extraction
python pipeline_main.py
```

