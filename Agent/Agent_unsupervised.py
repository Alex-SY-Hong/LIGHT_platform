import os
import subprocess
import sys
from pathlib import Path


def find_project_root(project_name: str = "LIGHT_platform-main") -> Path:
    """
    Walk upwards from current script location to find the project root folder.
    Works on both Windows and Linux.
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if p.name == project_name:
            return p
    # Fallback: adjust if your menu script lives inside the project
    # (e.g., LIGHT_platform-main/Unsupervised_Learning/run_menu.py)
    return here.parent.parent


def run_script(script_path: Path) -> bool:
    """Run the specified Python script (cross-platform)."""
    script_path = Path(script_path).resolve()

    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print(f"{'='*60}")

    if not script_path.exists():
        print(f"Error: File not found - {script_path}")
        return False

    try:
        # Set environment variables to ensure UTF-8 output and unbuffered mode
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        # Run inside the script's directory so relative paths in that script still work
        result = subprocess.run(
            [sys.executable, "-u", str(script_path.name)],
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )

        if result.stdout:
            print("Output:")
            print(result.stdout)

        if result.stderr:
            print("Errors:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"[SUCCESS] {script_path} executed successfully")
            return True
        else:
            print(f"[FAILED] {script_path} failed, return code: {result.returncode}")
            return False

    except Exception as e:
        print(f"Error occurred during execution: {e}")
        return False


def main():
    print("Starting the unsupervised learning automation runner...")

    project_root = find_project_root("LIGHT_platform-main")

    # Build paths relative to project root (cross-platform)
    scripts = [
        project_root / "Unsupervised_Learning" / "unsupervised_learning" / "unsupervised_learning_automation.py",
        project_root / "Unsupervised_Learning" / "candidate_umap" / "candidate_umap_automation.py",
    ]

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
            run_script(scripts[0])
        elif choice == "2":
            print("\nYou chose to run script 2 only...")
            run_script(scripts[1])
        elif choice == "3":
            print("\nYou chose to run script 1 and then script 2...")
            ok1 = run_script(scripts[0])
            if ok1:
                run_script(scripts[1])
        elif choice == "4":
            print("\nYou chose to run nothing.")
        elif choice == "5":
            print("\nExiting.")
            break
        else:
            print("\nInvalid choice. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
