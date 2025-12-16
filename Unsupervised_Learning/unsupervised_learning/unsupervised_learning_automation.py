import os
import re
import sys
import subprocess
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent  # this automation.py directory


def run_step(script_name, description, check_function=None, return_output=False):
    """
    Run a step script (located in BASE_DIR), optionally return stdout/stderr text.
    Cross-platform: Windows & Linux.
    """
    print(f"\n{'='*60}")
    print(f"Starting {description} ({script_name})...")
    print(f"{'='*60}")

    if check_function and check_function():
        print(f"[SKIP] {description} (output already exists)")
        return (True, "", "") if return_output else True

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return (False, "", f"Script not found: {script_path}") if return_output else False

    try:
        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            env=env,
            cwd=str(BASE_DIR),  # <--- lock working directory
        )

        if result.stdout and result.stdout.strip():
            print(f"\n--- STDOUT ({script_name}) ---")
            print(result.stdout.strip())

        if result.stderr and result.stderr.strip():
            print(f"\n--- STDERR ({script_name}) ---")
            print(result.stderr.strip())

        print(f"[OK] {description} completed successfully.")

        if check_function:
            if check_function():
                print("[OK] Validation passed")
            else:
                print("[ERROR] Validation failed")
                sys.exit(1)

        return (True, result.stdout, result.stderr) if return_output else True

    except subprocess.CalledProcessError as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        print(f"[ERROR] {description} failed (exit code {e.returncode})")

        if stdout.strip():
            print(f"\n--- STDOUT ({script_name}) ---")
            print(stdout.strip())
        if stderr.strip():
            print(f"\n--- STDERR ({script_name}) ---")
            print(stderr.strip())

        return (False, stdout, stderr) if return_output else False


def parse_best_cluster(text: str):
    if not text:
        return None

    patterns = [
        r"best\s*cluster\s*[:=]\s*(\d+)",
        r"best\s*classification.*?cluster\s*(\d+)",
        r"\bcluster[_\s-]?(\d+)\b",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def validate_data_process():
    return (BASE_DIR / "final_two_smiles_with_modulus.csv").exists()

def validate_morgan_generation():
    return (BASE_DIR / "AB_concat1024.npy").exists()

def validate_unsupervised():
    return (BASE_DIR / "cluster_umap_kmeans.png").exists()

def validate_analysis():
    return (BASE_DIR / "clusters" / "cluster_statistics.csv").exists()

def validate_umap_kmeans():
    return (BASE_DIR / "cluster_umap_kmeans_from_npy.png").exists()


if __name__ == "__main__":
    ok = run_step("data-process.py", "Data processing", validate_data_process)
    if not ok: sys.exit(1)

    ok = run_step("morgan.py", "Morgan fingerprint generation", validate_morgan_generation)
    if not ok: sys.exit(1)

    ok = run_step("unsupervised.py", "Unsupervised learning", validate_unsupervised)
    if not ok: sys.exit(1)

    ok = run_step("analyze_unsupervised.py", "Analysis of unsupervised results", validate_analysis)
    if not ok: sys.exit(1)

    ok = run_step("umap2d-kmeans.py", "UMAP + KMeans clustering", validate_umap_kmeans)
    if not ok: sys.exit(1)

    ok, out, err = run_step(
        "Best_classification_response.py",
        "API response and classification",
        check_function=None,
        return_output=True
    )
    if not ok:
        sys.exit(1)

    merged = (out or "") + "\n" + (err or "")
    best_n = parse_best_cluster(merged)

    if best_n is None:
        print("[ERROR] Could not parse best cluster id from output.")
        print("Please check Best_classification_response.py printed text format.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"[INFO] Parsed best cluster id = {best_n}")
    print(f"{'='*60}")

    cluster_csv = BASE_DIR / "clusters" / f"cluster_{best_n}.csv"
    if not cluster_csv.exists():
        print(f"[ERROR] File not found: {cluster_csv}")
        sys.exit(1)

    print(f"[INFO] Running: smiles_count_AB.py {cluster_csv}")
    subprocess.run(
        [sys.executable, "-u", str(BASE_DIR / "smiles_count_AB.py"), str(cluster_csv)],
        check=True,
        cwd=str(BASE_DIR),
    )

    produced_path = BASE_DIR / "clusters" / f"cluster_{best_n}-AB-unique.csv"
    if not produced_path.exists():
        alt = BASE_DIR / f"cluster_{best_n}-AB-unique.csv"
        if alt.exists():
            produced_path = alt
        else:
            print(f"[ERROR] Expected output not found: {produced_path}")
            sys.exit(1)

    dest_dir = (BASE_DIR / ".." / ".." / "Supervised_Learning" / "High-throughput predict").resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / "kmeans_results.csv"
    shutil.copy2(produced_path, dest_path)

    print(f"[OK] Copied & renamed:")
    print(f"     {produced_path}")
    print(f"  -> {dest_path}")

    print("\nPipeline execution completed successfully!")
