#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import subprocess
import sys
import os
from pathlib import Path


# ====================================================================
# --- Basic Utility Functions ---
# ====================================================================

def run_cmd(
    cmd: list[str],
    cwd: Path,
    check: bool = True,
    env_vars: dict = None,
    input_str: str = None  # Added to support auto-typing inputs
) -> subprocess.CompletedProcess | None:
    """
    Execute command and handle input/output streams.
    Can inject input (input_str) to automate interactive scripts.
    """
    if cmd and cmd[0] == "python":
        cmd = [sys.executable] + cmd[1:]
    elif cmd and cmd[0].endswith("python"):
        cmd[0] = sys.executable

    print("\n" + ">"*20 + " STARTING SUB-PROCESS " + "<"*20)
    print(f"📂 Working Directory: {cwd}")
    print(f"🚀 Executing Command: {' '.join(cmd)}")
    if input_str:
        # Visual debug of what we are typing into the process
        debug_input = input_str.replace("\n", " -> ")
        print(f"⌨️  Auto-Input Injection: {debug_input}")
    print("="*60 + "\n")

    # Prepare Environment
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if env_vars:
        env.update(env_vars)

    try:
        # We use input=input_str to simulate user typing
        # text=True ensures inputs/outputs are treated as strings
        p = subprocess.run(
            cmd, 
            cwd=str(cwd), 
            check=False, 
            env=env,
            input=input_str, 
            text=True 
        ) 
        
        if check and p.returncode != 0:
            print(f"⚠️ Sub-process returned non-zero exit code: {p.returncode}")
        return p
    except KeyboardInterrupt:
        print("\n⚠️ User interrupted the sub-process.")
        return None
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        return None


# ====================================================================
# --- 1. Unsupervised Learning Module (Sub-Menu) ---
# ====================================================================

def launch_unsupervised_pipeline(base_dir: Path, auto_mode: bool = False) -> None:
    """
    Handles Unsupervised Learning scripts.
    auto_mode=True:  Run Script 1 then Script 2 automatically.
    auto_mode=False: Show the specific 5-option interactive menu.
    """
    print("\n" + "*"*60)
    print("🧪 Entering [Unsupervised Learning] Module")
    print("*"*60)

    # Define Paths
    unsup_root = base_dir / "Unsupervised_Learning"
    script_algo = unsup_root / "unsupervised_learning" / "unsupervised_learning_automation.py"
    script_umap = unsup_root / "candidate_umap" / "candidate_umap_automation.py"

    # Helper to run a specific script
    def _run_script(script_path: Path):
        if not script_path.exists():
            print(f"❌ Error: File not found - {script_path}")
            return False
        res = run_cmd(["python", str(script_path.name)], cwd=script_path.parent)
        return res is not None and res.returncode == 0

    # --- AUTO MODE (For Main Menu Option 4) ---
    if auto_mode:
        print("⚙️ Auto-Mode detected: Running complete Unsupervised Pipeline...")
        if _run_script(script_algo):
            print("\n✅ Algorithm complete. Starting UMAP generation...")
            _run_script(script_umap)
        return

    # --- INTERACTIVE MODE (For Main Menu Option 1) ---
    while True:
        print("\nPlease choose an action:")
        print("1. Only run unsupervised learning")
        print("2. Candidate molecule distribution map (UMAP)")
        print("3. Run script 1 and then script 2 (Suggested)")
        print("4. Run nothing")
        print("5. Exit")  

        choice = input("\nEnter your choice (1/2/3/4/5): ").strip()

        if choice == "1":
            print("\nYou chose to run script 1 only...")
            _run_script(script_algo)

        elif choice == "2":
            print("\nYou chose to run script 2 only...")
            _run_script(script_umap)

        elif choice == "3":
            print("\nYou chose to run script 1 and then script 2...")
            ok1 = _run_script(script_algo)
            if ok1:
                _run_script(script_umap)

        elif choice == "4":
            print("\nYou chose to run nothing.")

        elif choice == "5":
            print("\nExiting Unsupervised Module -> Returning to Main Menu.")
            break 

        else:
            print("\nInvalid choice. Please enter 1, 2, 3, 4, or 5.")


# ====================================================================
# --- 2. Supervised Learning Module ---
# ====================================================================

def launch_supervised_pipeline(base_dir: Path, auto_mode: bool = False) -> None:
    """
    Launch the Supervised Learning Pipeline.
    auto_mode=True: Automatically simulates entering '3' then '4' to the subprocess.
    """
    agent_dir = base_dir / "Agent"
    script_path = agent_dir / "Supervised_pipline.py"

    # Logic to find the script if it's not in Agent folder
    if not script_path.exists():
        fallback_path = base_dir / "Supervised_Learning" / "Supervised_pipline.py"
        if fallback_path.exists():
            script_path = fallback_path
            run_cwd = base_dir / "Supervised_Learning"
        else:
            print(f"❌ Supervised Learning master script not found: {script_path}")
            return
    else:
        run_cwd = agent_dir

    print("\n" + "*"*60)
    print("🔬 Entering [Supervised Learning] Module")
    if auto_mode:
        print("⚙️ Auto-Mode detected: Triggering 'Run Both' then 'Exit'...")
    else:
        print("👉 Interactive Mode: Please operate in the sub-menu.")
    print("*"*60)

    if auto_mode:
        # --- FIXED HERE ---
        # "3\n" -> Selects 'Run Both'
        # "4\n" -> Selects 'Exit' after the tasks are done
        run_cmd(["python", str(script_path.name)], cwd=run_cwd, input_str="3\n4\n")
    else:
        # Interactive mode: No input_str, user types manually
        run_cmd(["python", str(script_path.name)], cwd=run_cwd)
    
    print("\n" + "*"*60)
    print("✅ [Supervised Learning] module task completed.")
    print("*"*60)


# ====================================================================
# --- 3. Agent Selection Module ---
# ====================================================================

def run_selection_pipeline(base_dir: Path) -> None:
    """Run the Agent Selection Script."""
    supervised_dir = base_dir / "Supervised_Learning"
    
    file_ym = supervised_dir / "Regression_Model" / "results" / "YoungsModulus" / "predictions" / "RF_best_pred_kmeans_results.csv"
    file_sr = supervised_dir / "Classification_Model" / "results" / "SwellingRatio" / "SwellingRatio_predict.csv"
    
    missing = []
    if not file_ym.exists(): missing.append("Regression Results (RF_best_pred_kmeans_results.csv)")
    if not file_sr.exists(): missing.append("Classification Results (SwellingRatio_predict.csv)")

    if missing:
        print("\n❌ [Agent Selection Aborted] Missing necessary files. Cannot run selection:")
        for m in missing:
            print(f"   - {m}")
        print("👉 Please ensure Supervised Learning models have generated predictions.")
        return

    print("\n=== [Task] Running Agent Selection Pipeline ===")
    agent_dir = base_dir / "Agent"
    selection_script = agent_dir / "Selection_pipeline.py"
    
    if not selection_script.exists():
        print(f"❌ Agent script not found: {selection_script}")
        return

    run_cmd(["python", str(selection_script.name)], cwd=agent_dir)


# ====================================================================
# --- 4. Run Full Workflow ---
# ====================================================================

def run_full_workflow(base_dir: Path) -> None:
    """Sequentially runs Unsupervised -> Supervised -> Selection"""
    print("\n" + "#"*60)
    print("🚀 STARTING FULL AUTOMATION WORKFLOW (1 -> 2 -> 3)")
    print("#"*60)

    # Step 1: Unsupervised (Auto: Runs script 1 then 2)
    launch_unsupervised_pipeline(base_dir, auto_mode=True)

    # Step 2: Supervised (Auto: Injects "3" then "4")
    print("\n>>> Proceeding to Step 2: Supervised Learning...")
    launch_supervised_pipeline(base_dir, auto_mode=True)

    # Step 3: Selection
    print("\n>>> Proceeding to Step 3: Agent Selection...")
    run_selection_pipeline(base_dir)

    print("\n" + "#"*60)
    print("🏁 FULL WORKFLOW COMPLETED")
    print("#"*60)


# ====================================================================
# --- Main Menu Logic ---
# ====================================================================

def prompt_main_menu() -> str:
    print("\n" + "="*60)
    print("🤖 LIGHT Platform Agent Workflow Controller")
    print("=" * 60)
    print("1. Run Unsupervised Learning (Opens Sub-menu)")
    print("2. Run Supervised Learning (Regression & Classification)")
    print("3. Run Agent Selection (Final Decision)")
    print("4. Run ALL (Sequence: 1 -> 2 -> 3)")
    print("5. Exit")
    print("=" * 60)
    return input("Please select (1-5): ").strip()


def main():
    ap = argparse.ArgumentParser(description="LIGHT Platform Agent Scheduler")
    args = ap.parse_args()

    # 1. Locate Root Directory
    current_path = Path(__file__).resolve()
    base_dir = current_path.parent.parent 
    
    if not (base_dir / "Agent").exists():
         print(f"❌ Path Error. Please check script location. Expected root: {base_dir}")
         sys.exit(1)

    print(f"📦 Project Root: {base_dir}")

    # 2. Main Loop
    while True:
        choice = prompt_main_menu()

        if choice == "5":
            print("👋 Program ended.")
            break

        try:
            if choice == "1":
                # Opens Detailed Menu
                launch_unsupervised_pipeline(base_dir, auto_mode=False)
            
            elif choice == "2":
                # Opens Interactive Supervised Menu
                launch_supervised_pipeline(base_dir, auto_mode=False)

            elif choice == "3":
                # Runs Selection Directly
                run_selection_pipeline(base_dir)

            elif choice == "4":
                # Runs Everything Automatically
                run_full_workflow(base_dir)

            else:
                print("❌ Invalid input. Please enter 1-5.")

        except Exception as e:
            print(f"\n❌ Unexpected Error: {e}")

if __name__ == "__main__":
    main()