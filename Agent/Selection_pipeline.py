import pandas as pd
import os
import sys

# --- Import LLM Decision and Analysis Modules ---
try:
    # Import LLM decision and analysis functions. Assuming they are in Selection_API.py.
    from Selection_API import get_merge_column_with_llm, analyze_and_explain_results_with_llm
except ImportError:
    # Critical Error: If the API module cannot be found, the program cannot start.
    print("❌ Error: Cannot import LLM functions from Selection_API.py. Please ensure Selection_API.py is in the LIGHT_platform-main/Agent/ directory.")
    sys.exit(1)


# --- Path Definition and Correction ---

# 1. Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Calculate project root directory (Go up one level to LIGHT_platform-main)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) 

# Print BASE_DIR for debugging to confirm it is the LIGHT_platform-main directory
print(f"DEBUG: Resolved Project Root (BASE_DIR): {BASE_DIR}")


# 💡 [Local Fallback Configuration] Hardcoded fallback column when LLM fails
LOCAL_FALLBACK_COLUMN = "Pair_ID" 

# --- File Path Definitions ---

FILE_PATH_1 = os.path.join(
    BASE_DIR, 
    "Supervised_Learning", 
    "Regression_Model", 
    "results", 
    "YoungsModulus", 
    "predictions", 
    "RF_best_pred_kmeans_results.csv"
)

FILE_PATH_2 = os.path.join(
    BASE_DIR, 
    "Supervised_Learning", 
    "Classification_Model", 
    "results", 
    "SwellingRatio", 
    "SwellingRatio_predict.csv"
)

# ================== Data Cleaning Function ==================

def clean_smiles_data(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """
    Remove rows containing specific duplicate SMILES patterns.
    """
    # Define SMILES patterns to remove
    SMILES_PATTERNS_TO_REMOVE = [
        r'O=C\(C\(C\[\*\]\)\s*\[\*\]\)N',
        r'\[\*\]OC1OC\(CO\)C\(OC2C\(O\)C\(O\)C\(\[\*\]\)C\(CO\)O2\)C\(O\)C1O'
    ]
    
    initial_count = len(df)
    mask = pd.Series([False] * initial_count, index=df.index)
    smiles_cols = [col for col in ['SMILE A', 'SMILE B'] if col in df.columns]

    if not smiles_cols:
        print(f"⚠️ Warning: '{df_name}' is missing 'SMILE A' or 'SMILE B' columns. Skipping SMILES cleaning.")
        return df

    for pattern in SMILES_PATTERNS_TO_REMOVE:
        for col in smiles_cols:
            mask |= df[col].astype(str).str.contains(pattern, na=False, regex=True)

    cleaned_df = df[~mask].copy()
    removed_count = initial_count - len(cleaned_df)
    if removed_count > 0:
        print(f"🧹 {df_name} SMILES cleaning complete: Removed {removed_count} rows of duplicate/invalid SMILES.")
    
    return cleaned_df

# ================== Manual Markdown Table Function (No Network Required) ==================
def dataframe_to_markdown(df):
    """
    Manually convert DataFrame to a Markdown table string.
    Solves the issue of missing 'tabulate' library in offline environments.
    """
    if df.empty:
        return ""
    try:
        # 1. Convert to string
        df_str = df.astype(str)
        # 2. Calculate column widths
        widths = []
        for col in df_str.columns:
            vals = df_str[col].values
            if len(vals) > 0:
                max_val = max(len(v) for v in vals)
            else:
                max_val = 0
            widths.append(max(len(str(col)), max_val))
        
        # 3. Construct rows
        header = "| " + " | ".join(f"{col:<{w}}" for col, w in zip(df_str.columns, widths)) + " |"
        separator = "| " + " | ".join("-" * w for w in widths) + " |"
        
        rows = []
        for _, row in df_str.iterrows():
            row_str = "| " + " | ".join(f"{val:<{w}}" for val, w in zip(row, widths)) + " |"
            rows.append(row_str)
            
        return "\n".join([header, separator] + rows)
    except Exception as e:
        return f"[Table display failed: {e}] (Data has been saved)"
# ======================================================================================

# --- Core Selection Logic (Including Merge) ---

def get_filtered_data(merge_column: str, output_filename="Agent_Final_Selection.csv"):
    """
    Read two CSV files, filter based on specific criteria, and perform an inner join merge based on a common column.
    """
    print("\n🚀 Starting data selection and merging process...")
    
    # 1. Read first file (YoungsModulus)
    try:
        df_ym = pd.read_csv(FILE_PATH_1)
        print(f"✅ Successfully read File 1: {os.path.basename(FILE_PATH_1)}, Rows: {len(df_ym)}")
    except FileNotFoundError:
        print(f"❌ Error: File 1 not found: {FILE_PATH_1}")
        return pd.DataFrame(), pd.DataFrame(), None 
    except Exception as e:
        print(f"❌ Failed to read File 1: {e}")
        return pd.DataFrame(), pd.DataFrame(), None

    # 2. Read second file (SwellingRatio)
    try:
        df_sr = pd.read_csv(FILE_PATH_2)
        print(f"✅ Successfully read File 2: {os.path.basename(FILE_PATH_2)}, Rows: {len(df_sr)}")
    except FileNotFoundError:
        print(f"❌ Error: File 2 not found: {FILE_PATH_2}")
        return pd.DataFrame(), pd.DataFrame(), None
    except Exception as e:
        print(f"❌ Failed to read File 2: {e}")
        return pd.DataFrame(), pd.DataFrame(), None

    # --- Data Cleaning: Remove Duplicate SMILES ---
    df_ym = clean_smiles_data(df_ym, "YoungsModulus Data")
    df_sr = clean_smiles_data(df_sr, "SwellingRatio Data")
    # --------------------------------

    # --- Filter Criteria ---
    COLUMN_YM = "RF_YoungsModulus_pred"
    MIN_YM = 2.0
    MAX_YM = 3.3
    COLUMN_SR = "Prediction"
    TARGET_SR = 1
    
    # A. YoungsModulus Filter
    if COLUMN_YM not in df_ym.columns:
        print(f"⚠️ Warning: File 1 is missing column '{COLUMN_YM}'. Skipping YM filtering.")
        filtered_ym = df_ym.copy()
    else:
        filtered_ym = df_ym[
            (df_ym[COLUMN_YM] >= MIN_YM) & 
            (df_ym[COLUMN_YM] <= MAX_YM)
        ].copy() 
        print(f"✨ File 1 filtered rows ({MIN_YM} <= {COLUMN_YM} <= {MAX_YM}): {len(filtered_ym)}")

    # B. SwellingRatio Filter
    if COLUMN_SR not in df_sr.columns:
        print(f"⚠️ Warning: File 2 is missing column '{COLUMN_SR}'. Skipping SR filtering.")
        filtered_sr = df_sr.copy()
    else:
        filtered_sr = df_sr[
            df_sr[COLUMN_SR] == TARGET_SR
        ].copy() 
        print(f"✨ File 2 filtered rows ({COLUMN_SR} == {TARGET_SR}): {len(filtered_sr)}")

    # 3. Inner Join Merge
    print(f"\n🤝 Attempting to merge results based on common column '{merge_column}'...")
    
    if merge_column not in filtered_ym.columns or merge_column not in filtered_sr.columns:
        print(f"❌ Error: Merge column '{merge_column}' is missing in at least one filtered dataset. Check LLM/Fallback configuration.")
        return filtered_ym, filtered_sr, None 

    merged_df = pd.merge(
        filtered_ym, 
        filtered_sr, 
        on=merge_column, 
        how='inner', 
        suffixes=('_YM', '_SR') 
    )
    
    print(f"🎉 Final merged row count (meeting all criteria): {len(merged_df)}")
    
    # 4. Organize Final Output Columns
    final_df = merged_df.rename(columns={
        'SMILE A_YM': 'SMILE A',
        'SMILE B_YM': 'SMILE B',
        'row_index_YM': 'row_index',
        COLUMN_YM + '_YM': COLUMN_YM, 
        COLUMN_SR + '_SR': COLUMN_SR  
    })
    
    # Define desired final column order
    final_cols_order = [
        merge_column, 
        'SMILE A', 
        'SMILE B', 
        'row_index', 
        COLUMN_YM, 
        COLUMN_SR, 
        'Prediction_prob_class0', 
        'Prediction_prob_class1'  
    ]
    
    # Select available columns
    available_cols = [c for c in final_cols_order if c in final_df.columns]
    final_df = final_df[available_cols]

    # 5. Save Final Merged Results to CSV
    output_dir = os.path.join(BASE_DIR, "Final_predction")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        final_df.to_csv(output_path, index=False)
        print(f"✅ Final merged results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save merged results: {e}")
        
    return filtered_ym, filtered_sr, final_df

# --- Execution and Output ---
if __name__ == '__main__':
    # 1. Pre-check
    if not os.path.isdir(BASE_DIR):
        print(f"❌ Critical Error: Project root {BASE_DIR} does not exist. Check script execution location.")
        sys.exit(1)

    # 2. **LLM Decision Layer: Determine COMMON_ID_COLUMN**
    
    # Temporarily read file headers to get column names
    try:
        df_ym_cols = pd.read_csv(FILE_PATH_1, nrows=0).columns.tolist()
        df_sr_cols = pd.read_csv(FILE_PATH_2, nrows=0).columns.tolist()
    except Exception as e:
        print(f"❌ Error: Cannot read file headers to determine column names: {e}")
        sys.exit(1)

    # Call LLM API (Auto fallback if network is down)
    try:
        current_merge_column = get_merge_column_with_llm(
            df1_columns=df_ym_cols, 
            df2_columns=df_sr_cols, 
            fallback_column=LOCAL_FALLBACK_COLUMN 
        )
    except Exception:
        print(f"⚠️ LLM Connection failed. Using fallback column: {LOCAL_FALLBACK_COLUMN}")
        current_merge_column = LOCAL_FALLBACK_COLUMN
    
    print(f"🤖 Flow determined merge column: {current_merge_column}")

    # 3. Run Data Processing Flow
    COLUMN_YM = "RF_YoungsModulus_pred"
    COLUMN_SR = "Prediction"
    
    df_ym_filtered, df_sr_filtered, df_merged = get_filtered_data(
        merge_column=current_merge_column, 
        output_filename="Agent_Final_Selection.csv"
    )
    
    # 4. Print Summary
    if df_ym_filtered is not None and df_sr_filtered is not None:
        
        # --- Deep Analysis & Explanation (LLM Intervention) ---
        structural_analysis = ""
        if df_merged is not None and not df_merged.empty:
            
            # Format final data using custom function (no tabulate required)
            final_data_markdown = dataframe_to_markdown(df_merged)
            
            youngs_range = f"2 - 3 (kPa, log10)"
            swelling_condition = f"Prediction Class 1"
            
            # Call LLM for analysis
            print("[LLM ANALYZE] Attempting to call LLM for structural analysis (skipping if offline)...")
            try:
                structural_analysis = analyze_and_explain_results_with_llm(
                    final_data_markdown, 
                    youngs_range, 
                    swelling_condition
                )
            except Exception as e:
                print(f"⚠️ Unable to connect to LLM for analysis: {e}")
                structural_analysis = "(LLM structural analysis skipped due to network/API issues)"
            
        
        print("\n" + "="*80)
        print("🤖 Decision Source: Hybrid Intelligence (API/Local Rules)") 
        print(f"📊 Flow Summary (Merge Column: {current_merge_column})")
        print("="*80)

        if df_merged is not None and not df_merged.empty:
            print(f"🔍 Final Merged Results ({len(df_merged)} rows, showing first 5):")
            
            display_cols_order = [
                current_merge_column, 'SMILE A', 'SMILE B', 'row_index', 
                COLUMN_YM, COLUMN_SR, 'Prediction_prob_class0', 'Prediction_prob_class1'
            ]
            # Filter safe columns to prevent display errors
            safe_display_cols = [c for c in display_cols_order if c in df_merged.columns]
            
            # Print using custom function
            print(dataframe_to_markdown(df_merged[safe_display_cols].head()))
                
        elif df_merged is not None and df_merged.empty:
            print("⚠️ Final merged result is empty. No samples satisfy both conditions simultaneously.")
        
        print(f"\n▶️ YoungsModulus Independent Filter Results: {len(df_ym_filtered)} rows")
        print(f"▶️ SwellingRatio Independent Filter Results: {len(df_sr_filtered)} rows")

        # Print LLM Deep Analysis Result
        if structural_analysis:
            print("\n" + "="*80)
            print("🧪 LLM Structure & Performance Analysis (Potential Reasons):")
            print("="*80)
            print(structural_analysis)
            print("="*80)