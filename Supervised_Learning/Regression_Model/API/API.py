#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# API.py

import os
import json
import openai

# ===== Basic Configuration: Recommended to use Environment Variables =====
# Example:
#   LLM_API_KEY   = Your API Key
#   LLM_BASE_URL  = https://your-proxy-domain/v1
#   LLM_MODEL_NAME= DeepSeek-R1  (or other model names)
LLM_API_KEY = os.getenv("LLM_API_KEY", "YOUR_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "YOUR_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "DeepSeek-R1")

# Initialize client (Create only once)
client = openai.OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

# ===== Optional: Simple connectivity test (Can be commented out if you don't want to test on every import) =====
if __name__ == "__main__":
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": "Hello world"}],
            temperature=0.0,
        )
        print("[TEST] LLM connection normal, response content:")
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"[TEST] LLM connection failed: {e}")


def _safe_parse_json_from_llm(text: str):
    """
    Attempt to parse JSON from the string returned by LLM,
    automatically removing potential ```json ... ``` wrappers.
    """
    s = text.strip()

    # Handle ```json ... ``` or ``` ... ``` format
    if s.startswith("```"):
        # Remove leading and trailing ```
        s = s.strip("`")
        # Remove potential json prefix
        if s.lower().startswith("json"):
            s = s[4:].lstrip()

    return json.loads(s)

def choose_best_with_llm(candidates_simple):
    """
    Use an LLM to select the best model from candidates_simple.
    If API fails, select the best model from local fallback.
    """
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
            timeout=10.0,
        )

        choices = getattr(resp, "choices", None)
        if not choices:
            raise ValueError(f"LLM returned no choices: {resp}")

        msg = choices[0].message
        llm_text = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Parse JSON safely
        best = _safe_parse_json_from_llm(llm_text)

        if not isinstance(best, dict) or "model_type" not in best or "best_fold" not in best:
            raise ValueError(f"Invalid JSON structure from LLM: {best}")

        best["api_status"] = "success"
        best["selection_source"] = "llm_api"
        best["selection_method"] = "llm_api_success"
        return best

    except Exception as e:
        # Stop successful logic execution on error, jump directly to fallback
        print(f"[WARN] LLM model selection failed: {e}")
        print("[INFO] Falling back to local max(best_score).")

        # Return local max score model directly
        best_local = max(candidates_simple, key=lambda x: x["best_score"])
        return {
            "model_type": best_local["model_type"],
            "best_fold": best_local["best_fold"],
            "best_score": best_local["best_score"],
            "api_status": "failed",  # Explicitly mark as failed
            "selection_source": "local_fallback",
            "selection_method": "local_fallback_after_api_failure",  # Explicitly mark local fallback
            "error": str(e)[:100]  # Limit error message length

        }
