#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# API.py

import os
import json
import openai
import traceback
import sys

# ===== Basic Configuration: Environment variables recommended =====
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-4x7BMQyP3IBNlnmOuOIRTg")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://llmapi.paratera.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "DeepSeek-R1")

# Initialize Client (Create only once)
try:
    client = openai.OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to initialize OpenAI Client: {e}")
    traceback.print_exc()
    sys.exit(1)

# Convention: Define path relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ===== Core Helper Functions (LLM related) =====

def _safe_parse_json_from_llm(text: str):
    """
    Attempt to parse JSON from a string returned by LLM, automatically removing potential ```json ... ``` wrappers.
    Now includes robust error handling for malformed JSON.
    """
    if not text:
        raise ValueError("LLM returned empty response.")

    s = text.strip()

    # Handle ```json ... ``` or ``` ... ``` format
    if "```" in s:
        # Split by ``` and take the content of the first block usually
        parts = s.split("```")
        # Usually the content is in the middle (index 1) if wrapped like ```json \n ... \n ```
        if len(parts) >= 3:
            s = parts[1]
            if s.lower().startswith("json"):
                s = s[4:].lstrip()
        else:
            # Fallback: simple strip if split didn't result in standard block
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:].lstrip()

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # Re-raise with the raw text context for debugging
        raise ValueError(f"JSON Decode Error: {e}\n[Raw Text caused error]: {text}")


# ===== Core Business Logic 1: Model Selection (LLM) =====

def choose_best_with_llm(candidates_simple):
    """
    Use an LLM to select the best model from candidates_simple, with local fallback.
    """
    if not candidates_simple:
        print("[WARN] No candidates provided to choose_best_with_llm.")
        return {}

    try:
        # Construct prompt
        c_json = json.dumps(candidates_simple, ensure_ascii=False, indent=2)
        prompt = f"""
        You are a machine-learning assistant. Your task is to select the best model based on cross-validation results.

        Each item in the list is a dictionary containing:
        - model_type: the model category (rf / mlp / svm / ols)
        - best_fold: the fold number that achieved the best score within that model
        - best_score: the R² value for that best fold

        Your job:
        - Compare all models by best_score (higher is better)
        - Select the single model with the highest best_score
        - Return ONLY a strict JSON object with the following structure, and no explanations or extra text:

        {{
          "model_type": "rf",
          "best_fold": 8,
          "best_score": 0.6384
        }}

        Here is the list of candidate models:
        {c_json}
        """.strip()

        # Call LLM API
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=15.0, # Slight increase in timeout
        )

        choices = getattr(resp, "choices", None)
        if not choices:
            raise ValueError(f"LLM returned no choices. Full Response: {resp}")

        msg = choices[0].message
        llm_text = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Parse JSON safely
        best = _safe_parse_json_from_llm(llm_text)

        if not isinstance(best, dict) or "model_type" not in best or "best_fold" not in best:
            raise ValueError(f"Invalid JSON structure (missing keys) from LLM: {best}")

        best["api_status"] = "success"
        best["selection_source"] = "llm_api"
        best["selection_method"] = "llm_api_success"
        return best

    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        print(f"[ERROR] LLM Network/Timeout Error: {e}")
        print("[INFO] Falling back to local logic due to network failure.")
    except ValueError as e:
        print(f"[ERROR] LLM Parsing/Logic Error: {e}")
        print("[INFO] Falling back to local logic due to parsing failure.")
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"[ERROR] Unexpected error in choose_best_with_llm: {e}")
        traceback.print_exc() # Print full stack trace for debugging

    # --- FALLBACK LOGIC ---
    try:
        best_local = max(candidates_simple, key=lambda x: x.get("best_score", -float('inf')))
        return {
            "model_type": best_local["model_type"],
            "best_fold": best_local["best_fold"],
            "best_score": best_local["best_score"],
            "api_status": "failed",
            "selection_source": "local_fallback",
            "selection_method": "local_fallback_after_api_failure",
            "error": "See logs for details"
        }
    except Exception as fallback_e:
        print(f"[CRITICAL] Local fallback also failed: {fallback_e}")
        return {"error": "Critical failure in both LLM and Fallback"}


# ===== Core Business Logic 2: Intelligent Merge Key Identification =====

def get_merge_column_with_llm(df1_columns: list, df2_columns: list, fallback_column: str = "Pair_ID") -> str:
    """
    Use LLM to infer the merge column, falling back to a specified column on failure.
    """
    # Pre-validation: Check if columns are empty
    if not df1_columns or not df2_columns:
        print("[WARN] One or both DataFrame column lists are empty. Using fallback.")
        return fallback_column

    column_list_1 = json.dumps(df1_columns, ensure_ascii=False)
    column_list_2 = json.dumps(df2_columns, ensure_ascii=False)
    
    prompt = f"""
    You are a data processing assistant. Your task is to identify the common unique identifier column
    (key) needed to perform an INNER JOIN between two datasets.

    The first dataset columns: {column_list_1}
    The second dataset columns: {column_list_2}
    
    Your job:
    - Analyze the column names.
    - Identify the single column name that serves as the unique identifier (must be in both lists).
    - Return ONLY the exact column name as a single, unquoted string.
    
    Example response: Pair_ID
    """.strip()

    try:
        # 1. Call LLM API
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=10.0,
        )

        # 2. Parse and clean LLM output
        llm_content = resp.choices[0].message.content
        if not llm_content:
            raise ValueError("Empty response content from LLM")
            
        llm_column = llm_content.strip().strip('"').strip("'").split('\n')[0].strip() # Handle cases where LLM adds newlines
        
        # 3. Validate LLM returned column name
        if llm_column in df1_columns and llm_column in df2_columns:
            print(f"[LLM SUCCESS] Intelligent merge column identification successful: {llm_column}")
            return llm_column
        else:
            # Detailed error message regarding why validation failed
            missing_in = []
            if llm_column not in df1_columns: missing_in.append("Dataset 1")
            if llm_column not in df2_columns: missing_in.append("Dataset 2")
            raise ValueError(f"LLM suggested '{llm_column}', but it is missing in: {', '.join(missing_in)}")

    except Exception as e:
        print(f"[WARN] LLM merge column identification failed: {e}")
        # Only print traceback if it's not a simple validation error we raised ourselves
        if not isinstance(e, ValueError):
            traceback.print_exc()
        
        # Local fallback logic
        if fallback_column in df1_columns and fallback_column in df2_columns:
            print(f"[LOCAL FALLBACK SUCCESS] Fallback common column found: {fallback_column}")
            return fallback_column
        
        print(f"[LOCAL FALLBACK FAILED] Default column '{fallback_column}' not found in both datasets.")
        return fallback_column 


# ===== Core Business Logic 3: Result Analysis and Explanation =====

def analyze_and_explain_results_with_llm(final_df: str, youngs_range: str, swelling_condition: str) -> str:
    """
    Use LLM to analyze filtered material data.
    """
    if not final_df or len(final_df) < 10:
        return "❌ Data insufficient for analysis (Empty or too short)."

    # Construct prompt
    prompt = f"""
    You are a materials science and chemistry expert. Analyze this dataset:
    
    **Criteria:** Young's: {youngs_range}, Swelling: {swelling_condition}

    **Data (Markdown/Table):**
    {final_df}

    Your job:
    1. **Analyze SMILE A and SMILE B:** Identify common functional groups/structures.
    2. **Connect Structure to Performance:** Why do these achieve the targets?
    3. **Conclusion:** Structural hypothesis for future design.

    Return clear, professional text.
    """.strip()

    try:
        print("\n[LLM ANALYZE] Calling LLM for deep chemical/structural analysis...")
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=60.0,
        )

        analysis_text = resp.choices[0].message.content
        if not analysis_text:
            return "❌ LLM returned empty analysis text."
            
        print("[LLM ANALYZE SUCCESS] Analysis complete.")
        return analysis_text

    except openai.APITimeoutError:
        print("[LLM ANALYZE FAILED] Request timed out (Wait > 60s).")
        return "❌ Analysis failed: The AI model took too long to respond."
    except openai.APIConnectionError:
        print("[LLM ANALYZE FAILED] Network connection error.")
        return "❌ Analysis failed: Network/API connection issue."
    except Exception as e:
        print(f"[LLM ANALYZE FAILED] Deep analysis failed: {e}")
        traceback.print_exc()
        return f"❌ Deep analysis failed due to internal error: {str(e)[:100]}..."

# ===== Optional: Simple Connectivity Test =====
if __name__ == "__main__":
    print("\n" + "="*60)
    print("API.py Standalone Run Test & Error Check")
    print("="*60)
    
    # 1. LLM Connection Test
    try:
        print("[TEST] Attempting simple handshake with LLM...")
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.0,
            timeout=5.0
        )
        print("[TEST] LLM Connection Normal.")
    except openai.AuthenticationError:
        print("[TEST FAILED] Invalid API Key. Please check 'LLM_API_KEY'.")
    except openai.APITimeoutError:
        print("[TEST FAILED] Connection timed out. Check your network or base URL.")
    except Exception as e:
        print(f"[TEST FAILED] Connection Error: {e}")
        traceback.print_exc()
    
    print("="*60)