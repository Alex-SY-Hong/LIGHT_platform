import os
import subprocess
import sys

def run_step(script_name, description, check_function=None):
    """
    Function to run each step of the pipeline.
    :param script_name: Script name to execute (without the .py extension)
    :param description: Short description for the step
    :param check_function: Optional function to validate the output
    """
    print(f"============================================================")
    print(f"Starting {description}...")
    print(f"============================================================")

    # Check if the output file exists before running the script
    if check_function and check_function():
        print(f"✓ Skipping {description} because the required file already exists.")
        return  # Skip the step if the file already exists

    try:
        # Run the script
        subprocess.run(['python', script_name], check=True)
        print(f"{description} completed successfully.")

        # If a check function is provided, run it to validate the output
        if check_function:
            if check_function():
                print(f"Validation for {description} passed.")
            else:
                print(f"Validation for {description} failed.")
                sys.exit(1)
        else:
            print(f"No validation needed for {description}.")

    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        sys.exit(1)

# Define validation functions for each step
def validate_morgan_fingerprints():
    # Check if the generated Morgan fingerprints file exists
    return os.path.exists('all_random_smiles_AB_concat1024.npy')

def validate_candidate():
    # Check if the candidate SMILES file exists
    return os.path.exists('Prediction-1028-ALL2-candidate-1024.npy')

def validate_umap_candidate():
    # Check if the UMAP result file exists
    return os.path.exists('umap_candidate_visualization.png')

if __name__ == "__main__":
    # Run each step and validate, skipping if the corresponding file exists
    run_step('all_AB_smiles2morgan.py', 'Count all SMILES and generate Morgan fingerprints', validate_morgan_fingerprints)
    run_step('cluster-3-AB-morgan.py', 'Generate Morgan fingerprints for SMILES pairs', validate_candidate)
    run_step('umap-candidate.py', 'Generate UMAP distribution plot for candidate components', validate_umap_candidate)

    print("Pipeline execution completed successfully!")
