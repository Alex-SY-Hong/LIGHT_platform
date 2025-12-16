#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# API.py

import os
import json
import openai
import pandas as pd
import sys

# ===== Basic Configuration =====
LLM_API_KEY = os.getenv("LLM_API_KEY", "YOUR_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "YOUR_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "DeepSeek-R1")


# Initialize client
client = openai.OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

def _safe_parse_json_from_llm(text: str):
    """
    Safely parse JSON string returned by LLM
    """
    s = text.strip()

    # Handle ```json ... ``` or ``` ... ``` format
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()

    return json.loads(s)

def get_best_cluster_from_csv(csv_path):
    """
    Read the cluster statistics CSV file, use LLM API to compare prob_SR and prob_YM,
    and return the best cluster_file along with its related probabilities.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_columns = ['prob_SR', 'prob_YM', 'SR_ge_9', 'YM_100_2000', 'cluster_file']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"[ERROR] CSV file is missing required columns: {', '.join(missing_columns)}")
        sys.exit(1)

    # Prepare candidate data for LLM selection
    candidates = []
    for idx, row in df.iterrows():
        candidate = {
            "cluster_file": row['cluster_file'],
            "prob_SR": float(row['prob_SR']),
            "prob_YM": float(row['prob_YM']),
            "SR_ge_9": int(row['SR_ge_9']),
            "YM_100_2000": int(row['YM_100_2000'])
        }
        candidates.append(candidate)
    
    # Use LLM API to select the best cluster
    try:
        # Construct prompt
        c_json = json.dumps(candidates, ensure_ascii=False, indent=2)
        prompt = f"""
        You are a materials science expert. Your task is to select the best hydrogel cluster based on probability analysis.
        
        Each item in the list is a dictionary containing:
        - cluster_file: the cluster file name
        - prob_SR: probability of having high stretchability (SR ≥ 9)
        - prob_YM: probability of having suitable Young's modulus (100-2000 MPa)
        - SR_ge_9: number of samples in cluster with SR ≥ 9
        - YM_100_2000: number of samples in cluster with YM between 100-2000 MPa
        
        Your task:
        - Analyze both prob_SR and prob_YM values
        - Consider the balance between stretchability and stiffness
        - Select the single cluster that has the best overall probability distribution
        - Return ONLY a strict JSON object, no explanations or extra text:
        
        {{
          "cluster_file": "cluster_1.csv",
          "prob_SR": 0.85,
          "prob_YM": 0.78,
          "SR_ge_9": 15,
          "YM_100_2000": 12,
          "selection_reason": "Because the Young's modulus and the water absorption expansion rate have the highest possibility of forming the target value"
        }}
        
        Candidate clusters list:
        {c_json}
        """.strip()
        
        # Call LLM API
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=60.0,
        )
        
        choices = getattr(resp, "choices", None)
        if not choices:
            raise ValueError(f"LLM returned no selection results: {resp}")
            
        msg = choices[0].message
        llm_text = msg.content if isinstance(msg.content, str) else str(msg.content)
        
        # Safely parse JSON
        best = _safe_parse_json_from_llm(llm_text)
        
        if not isinstance(best, dict) or "cluster_file" not in best:
            raise ValueError(f"Invalid JSON structure returned by LLM: {best}")
            
        best["api_status"] = "success"
        best["selection_source"] = "llm_api"
        best["selection_method"] = "llm_api_success"
        
        # Get the best row data
        best_row = df[df['cluster_file'] == best['cluster_file']].iloc[0]
        
        return {
            'cluster_file': best['cluster_file'],
            'prob_SR': float(best_row['prob_SR']),
            'prob_YM': float(best_row['prob_YM']),
            'SR_ge_9': int(best_row['SR_ge_9']),
            'YM_100_2000': int(best_row['YM_100_2000']),
            'selection_reason': best.get('selection_reason', 'Selected by LLM'),
            'api_status': best.get('api_status', 'unknown'),
            'selection_source': best.get('selection_source', 'unknown')
        }
        
    except Exception as e:
        # Fall back to local logic when API fails
        print(f"[WARN] LLM cluster selection failed: {e}")
        print("[INFO] Falling back to local maximum probability selection.")
        
        # Use local maximum probability selection
        df['max_prob'] = df[['prob_SR', 'prob_YM']].max(axis=1)
        best_row = df.loc[df['max_prob'].idxmax()]
        
        return {
            'cluster_file': best_row['cluster_file'],
            'prob_SR': float(best_row['prob_SR']),
            'prob_YM': float(best_row['prob_YM']),
            'SR_ge_9': int(best_row['SR_ge_9']),
            'YM_100_2000': int(best_row['YM_100_2000']),
            'selection_reason': 'The most likely classification for the target property of the hydrogel is:',
            'api_status': 'failed',
            'selection_source': 'local_fallback',
            'error': str(e)[:100]
        }
