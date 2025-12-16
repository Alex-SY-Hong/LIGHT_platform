#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local-Only Script: Use LLM API to select the "Recommended Best Model" from candidate models.

Usage Examples:

    # Run model recommendation only
    python advise_best_model_with_api.py \
        --candidates_json "results/YoungsModulus/model_candidates_for_llm.json" \
        --out_json "results/YoungsModulus/overall_best_model.json"

    # Run regression analysis first, then run model recommendation
    python advise_best_model_with_api.py \
        --candidates_json "results/YoungsModulus/model_candidates_for_llm.json" \
        --out_json "results/YoungsModulus/overall_best_model.json" \
        --run_regression_first
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from API import choose_best_with_llm  # The function you already have


def select_best_locally(candidates):
    """Select best model locally: Choose the model with the highest best_score."""
    if not candidates:
        return None
    
    best = max(candidates, key=lambda x: x.get("best_score", 0))
    return {
        "model_type": best["model_type"],
        "best_fold": best["best_fold"],
        "best_score": best["best_score"],
        "selection_method": "local_fallback_after_api_failure"  # Set local fallback flag
    }

def main():
    parser = argparse.ArgumentParser(description="Use LLM to advise on the best model among candidates")
    parser.add_argument(
        "--candidates_json",
        required=True,
        help="model_candidates_for_llm.json exported by Linux main.py",
    )
    parser.add_argument(
        "--out_json",
        help="Optional: Write the result selected by LLM to JSON (e.g., overall_best_model.json)",
    )
    parser.add_argument(
        "--run_regression_first",
        action="store_true",
        help="Run regression analysis first to generate candidate models",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts for API call, default is 3",
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=5,
        help="Retry delay time (seconds), default is 5 seconds",
    )
    parser.add_argument(
        "--skip_on_api_failure",
        action="store_true",
        help="Skip model recommendation and do not output results if API call fails",
    )
    parser.add_argument(
        "--use_local_fallback",
        action="store_true",
        help="Use local fallback strategy if API call fails",
    )
    
    args = parser.parse_args()
    
    # If regression analysis needs to be run first
    if args.run_regression_first:
        print("[INFO] Running regression analysis first...")
        regression_cmd = [
            "python", "regression_main.py",
            "--raw_csv", "../DataBase/youngs_modulus.csv",
            "--target", "Young's Modulus (kPa) log10",
            "--out_root", "results/YoungsModulus",
            "--polymer_cols", "SMILE A", "SMILE B", "SMILE C",
            "--do_predict",
            "--predict_in_csv", "../High-throughput predict/kmeans-pooled.csv",
            "--predict_source_csv", "../High-throughput predict/kmeans_results.csv",
            "--predict_target_name", "YoungsModulus_pred"
        ]
        
        result = subprocess.run(regression_cmd)
        if result.returncode != 0:
            print("[ERROR] Regression analysis failed")
            sys.exit(1)
    
    # Check if candidate models file exists
    candidates_path = Path(args.candidates_json).resolve()
    if not candidates_path.is_file():
        print(f"[ERROR] Candidate list JSON file not found: {candidates_path}")
        sys.exit(1)
    
    # Check if file is empty
    if candidates_path.stat().st_size == 0:
        print(f"[ERROR] Candidate list JSON file is empty: {candidates_path}")
        sys.exit(1)
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    print(f"[INFO] Found {len(candidates)} candidate models:")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c}")
    
    # Initialize result variables
    best = None
    api_success = False
    
    # Attempt to call API, supporting retries
    for attempt in range(args.max_retries):
        try:
            print(f"\n[INFO] Attempt {attempt + 1} to call LLM API...")
            best = choose_best_with_llm(candidates)
            
            # Modification: Determine success based on api_status or selection_source
            if best.get("api_status") == "success" or best.get("selection_source") == "llm_api":
                # API call successful
                api_success = True
                # Set this flag only when API succeeds
                if "selection_method" in best:
                    # Modification: Flag including attempt count
                    best["selection_method"] = f"llm_api_success_attempt_{attempt + 1}"
                break
            else:
                # This is a local fallback result, not API success
                print(f"[WARN] Attempt {attempt + 1} returned local fallback result, retrying...")
                # Continue retrying
                continue
                
        except Exception as e:
            print(f"[WARN] Attempt {attempt + 1} failed: {str(e)}")
            if attempt < args.max_retries - 1:
                print(f"[INFO] Retrying in {args.retry_delay} seconds...")
                time.sleep(args.retry_delay)
            else:
              print(f"[ERROR] All {args.max_retries} API call attempts failed")
    
    # Handle API call failure
    if not api_success:
      # Output results (using whatever fallback 'best' currently holds if choose_best_with_llm returns a fallback)
      print("\n" + "="*60)
      print("[RESULT] Model Recommendation Result:")
      print("="*60)
      if best is None and args.use_local_fallback:
          best = select_best_locally(candidates)
      
      if best:
          for key, value in best.items():
              print(f"  {key}: {value}")
          
          # Modification here: Output info based on api_status
          api_status = best.get("api_status", "unknown")
          if api_status == "success":
               # This branch is theoretically unreachable if api_success is False, but keeping logic parallel
              print(f"\n[INFO]  Model recommended by LLM API")
          else:
              print(f"\n[WARNING]  Model selected by local fallback strategy (API call failed)")
              print(f"  - Failure Reason: {best.get('error', 'Unknown Error')}")
              print(f"  - Selection Strategy: Choose model with highest best_score ({best.get('best_score', 'N/A')})")
      else:
          # If best is still None (e.g. no candidates or no fallback allowed)
          pass # Will be caught by the check below
    
    # Check if there is a best model result
    if not best or "model_type" not in best:
        print("[ERROR] Unable to generate best model recommendation")
        sys.exit(1)
    
    # Output results
    print("\n" + "="*60)
    print("[RESULT] Model Recommendation Result:")
    print("="*60)
    for key, value in best.items():
        print(f"  {key}: {value}")
    
    # Display selection method explanation
    method = best.get("selection_method", "unknown")
    if "llm_api_success" in method:
        print(f"\n[INFO]  Model recommended by LLM API")
    elif "local_fallback" in method:
        print(f"\n[WARNING]  Model selected by local fallback strategy (API call failed)")
        print("  - Local Strategy: Choose model with highest best_score")
    
    # If out_json is provided, write to file
    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        
        if api_success:
            print(f"\n[SUCCESS]  LLM recommended best model written to: {out_path}")
        else:
            print(f"\n[WARNING]  Local fallback best model written to: {out_path}")
            print("  [Note] This is not an LLM recommendation, but a fallback choice after API failure")
    
    print("\n" + "="*60)
    if api_success:
        print("[SUCCESS]  LLM Model Recommendation Completed")
    else:
        print("[WARNING]  Local Fallback Selection Completed (LLM API Call Failed)")
    print("="*60)


if __name__ == "__main__":
    main()