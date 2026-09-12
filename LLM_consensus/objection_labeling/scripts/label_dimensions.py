"""
Label anti-formula objections with dimension tags (objection_labeling module).

LLM-assisted multi-label annotation of anti-arguments against Formulas 4
and 5, scoped to the target formula. Outputs labeled arguments with
multi-hot vectors, co-occurrence statistics, and network graphs.

Dimensions (7 = 6 scoring dimensions + Other):
1. Mechanical_Safety
2. Swelling_Performance
3. Endothelialization
4. SMC_inhibition
5. Anti_inflammation
6. Thrombogenicity
7. Other (issues not fitting above dimensions)

Inputs : objection_labeling/data/filtered_anti_formula_{4,5}_from_raw.json
Outputs: objection_labeling/results/
Requires .env at repo root with API_KEY / API_URL.
"""

import json
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Package import (single source of truth for DIMENSIONS / vector encoding)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from objection_labeling.analysis.multihot import DIMENSIONS, labels_to_vector

# Load API credentials from repo root .env
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Configuration
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
LABEL_MODEL = "gemini-3.1-flash-lite-preview"

DIMENSION_DESCRIPTIONS = {
    "Mechanical_Safety": "Issues related to mechanical properties, modulus, strength, durability, structural integrity",
    "Swelling_Performance": "Issues related to water absorption, swelling ratio, conformability, lumen occlusion risk",
    "Endothelialization": "Issues related to endothelial cell attachment, healing, re-endothelialization, biointegration",
    "SMC_inhibition": "Issues related to smooth muscle cell suppression, neointimal hyperplasia, restenosis prevention",
    "Anti_inflammation": "Issues related to inflammatory response, immune reaction, biocompatibility",
    "Thrombogenicity": "Issues related to blood clotting, platelet adhesion, thrombosis, hemocompatibility",
    "Other": "Issues not fitting the above six dimensions"
}

# Target formula identity for scoped labeling (compositions from prompt.md)
FORMULA_DESCRIPTIONS = {
    4: "Formula 4: Gelatin_methacrylate (GelMA) & Silk_Fibroin (SK)",
    5: "Formula 5: Gelatin_methacrylate (GelMA) & Polyethylene_glycol (PEG)"
}


def save_multihot_tables(f4_labeled, f5_labeled, output_dir):
    """Write multi-hot encodings as JSON (with dimension order) and CSV matrix."""
    out = {"dimension_order": DIMENSIONS, "formula_4": [], "formula_5": []}
    csv_rows = []

    for formula, labeled in (("formula_4", f4_labeled), ("formula_5", f5_labeled)):
        for result in labeled:
            vector = labels_to_vector(result['labels'])
            out[formula].append({"key": result['key'], "vector": vector})
            csv_rows.append([formula, result['key']] + vector)

    json_path = output_dir / "multihot_vectors.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "multihot_vectors.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["formula", "key"] + DIMENSIONS)
        writer.writerows(csv_rows)

    print(f"Multi-hot vectors saved to: {json_path}")
    print(f"Multi-hot matrix saved to: {csv_path}")


def make_api_client(api_key: str, base_url: str):
    """Create OpenAI client for API calls."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


def label_argument(client, argument_text: str, formula_desc: str):
    """
    Label an anti-argument with negative/constraining dimension(s) using AI,
    scoped to a single target formula.
    Returns: list of dimension names
    """
    # Format dimension list for prompt
    dim_list = "\n".join([f"{i+1}. {dim}: {DIMENSION_DESCRIPTIONS[dim]}" for i, dim in enumerate(DIMENSIONS)])

    prompt = f"""You are analyzing anti-arguments against biomaterial formulas for vascular grafts.

TARGET FORMULA: {formula_desc}

This argument evaluates MULTIPLE candidate formulas and may mention their pros and cons. You must focus ONLY on the TARGET FORMULA. Ignore every statement about other formulas, even if they describe serious flaws.

Extract ONLY the dimension(s) that have NEGATIVE effects or CONSTRAINTS on the TARGET FORMULA as stated in the argument.

Rules:
- If the text says "A and B are good, BUT C limits performance", return ONLY "C"
- If the text mentions "the target formula has risk X", return the dimension of X
- If the text says "despite good A, B is insufficient" about the target formula, return "B"
- Extract dimensions described as problems, limitations, risks, constraints, or deficiencies OF THE TARGET FORMULA
- DO NOT include dimensions that are described positively (good, excellent, strong) unless they have a caveat/limitation
- DO NOT extract dimensions from flaws attributed to OTHER formulas (e.g., "Formula 2 has cytotoxicity risk" must NOT be labeled when the target is Formula 4)
- A general comparative statement like "left short of the winner's balanced profile" without a specific target-formula flaw maps to "Other"
- Dimensions outside the six listed (e.g., processing complexity, degradation timeline, contamination) map to "Other"

Dimensions:
{dim_list}

Argument: "{argument_text}"

Output format: Return ONLY the dimension numbers separated by commas (e.g., "1,3" or "2" or "6,7").
Return ONLY the negative/constraining dimensions OF THE TARGET FORMULA. If all dimensions mentioned are positive, return the most limiting one.
Do NOT include explanations."""

    try:
        response = client.chat.completions.create(
            model=LABEL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()

        # Parse result - should be comma-separated numbers
        labels = []
        for item in result.split(','):
            item = item.strip()
            if item.isdigit():
                idx = int(item) - 1  # Convert to 0-based
                if 0 <= idx < len(DIMENSIONS):
                    labels.append(DIMENSIONS[idx])
                else:
                    print(f"    [Warning] Invalid dimension number: {item}")
            elif item.lower() == 'other':
                labels.append("Other")

        if not labels:
            print(f"    [Warning] No valid labels parsed from: {result}")
            return ["Other"]  # Default fallback

        return labels

    except Exception as e:
        print(f"    [Error] API call failed: {e}")
        return ["Other"]  # Default fallback


def label_arguments_batch(client, arguments_dict, formula_id):
    """
    Label all arguments for a formula.
    arguments_dict: {"model Run X": ["arg1", "arg2", ...], ...}
    Returns: list of dicts with argument info and labels
    """
    results = []
    total = sum(len(args) for args in arguments_dict.values())
    formula_desc = FORMULA_DESCRIPTIONS[formula_id]

    print(f"\nLabeling Formula {formula_id} ({total} arguments) [scoped: {formula_desc}]...")

    for key, arg_list in arguments_dict.items():
        for i, arg_text in enumerate(arg_list):
            print(f"  [{key} arg {i+1}]...")
            labels = label_argument(client, arg_text, formula_desc)

            results.append({
                'key': key,
                'argument': arg_text,
                'labels': labels
            })

            print(f"    → {', '.join(labels)}")

    return results


def build_co_occurrence_matrix(labeled_results):
    """
    Build dimension co-occurrence matrix for a single formula.
    Returns: dict mapping (dim1, dim2) -> co_occurrence count
    """
    # Initialize matrix
    co_occurrence = defaultdict(int)
    dimension_counts = defaultdict(int)

    for result in labeled_results:
        labels = result['labels']
        # Count individual dimensions
        for dim in labels:
            dimension_counts[dim] += 1

        # Count co-occurrences
        for i, dim1 in enumerate(labels):
            for dim2 in labels[i+1:]:
                # Sort pair to avoid double counting
                pair = tuple(sorted([dim1, dim2]))
                co_occurrence[pair] += 1

    return co_occurrence, dimension_counts


def create_network_graph(co_occurrence, dimension_counts, formula_id, output_path):
    """
    Create and save network visualization graph for a specific formula.
    """
    # Create graph
    G = nx.Graph()

    # Add nodes (dimensions)
    for dim in DIMENSIONS:
        if dim in dimension_counts:
            G.add_node(dim, weight=dimension_counts[dim])

    # Add edges (co-occurrences)
    max_weight = max(co_occurrence.values()) if co_occurrence else 1

    for (dim1, dim2), count in co_occurrence.items():
        # Normalize edge weight for visualization
        weight = count / max_weight
        G.add_edge(dim1, dim2, weight=count, normalized_weight=weight)

    # Create visualization
    plt.figure(figsize=(12, 10))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node sizes based on occurrence count (only dims present in graph)
    node_sizes = [dimension_counts[dim] * 500 for dim in G.nodes()]

    # Edge widths based on co-occurrence
    edge_widths = []
    for dim1, dim2 in G.edges():
        weight = G[dim1][dim2]['normalized_weight']
        edge_widths.append(weight * 5)  # Scale for visibility

    # Color mapping (only dims present in graph)
    node_colors = [plt.cm.Set3(DIMENSIONS.index(dim) / len(DIMENSIONS)) for dim in G.nodes()]

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.9, edgecolors='gray', linewidths=2)

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    plt.title(f"Formula {formula_id} - Dimension Co-occurrence Network\n(Node size = occurrence count, Edge width = co-occurrence)",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Network graph saved to: {output_path}")


def create_statistics_summary(f4_labeled, f5_labeled, dimension_counts, co_occurrence, output_path):
    """
    Create statistics summary document with formula-specific layered statistics.
    """
    summary = []

    summary.append("# Dimension Labeling Statistics (Negative Effects Only)\n")
    summary.append("## Overview\n")
    summary.append(f"Formula 4 arguments: {len(f4_labeled)}\n")
    summary.append(f"Formula 5 arguments: {len(f5_labeled)}\n")
    summary.append(f"Total arguments: {len(f4_labeled) + len(f5_labeled)}\n")
    summary.append(f"Labeling: Only negative/constraining dimensions extracted\n")

    # Formula 4 Statistics
    f4_dim_counts = defaultdict(int)
    for result in f4_labeled:
        for dim in result['labels']:
            f4_dim_counts[dim] += 1

    f4_total = len(f4_labeled)  # total arguments, not total label instances

    summary.append("\n## Formula 4 - Dimension Occurrence\n")
    summary.append("| Dimension | Count | Percentage |\n")
    summary.append("|----------|-------|------------|\n")

    for dim in DIMENSIONS:
        count = f4_dim_counts.get(dim, 0)
        pct = (count / f4_total * 100) if f4_total > 0 else 0
        summary.append(f"| {dim} | {count} | {pct:.1f}% |\n")

    # Formula 5 Statistics
    f5_dim_counts = defaultdict(int)
    for result in f5_labeled:
        for dim in result['labels']:
            f5_dim_counts[dim] += 1

    f5_total = len(f5_labeled)  # total arguments, not total label instances

    summary.append("\n## Formula 5 - Dimension Occurrence\n")
    summary.append("| Dimension | Count | Percentage |\n")
    summary.append("|----------|-------|------------|\n")

    for dim in DIMENSIONS:
        count = f5_dim_counts.get(dim, 0)
        pct = (count / f5_total * 100) if f5_total > 0 else 0
        summary.append(f"| {dim} | {count} | {pct:.1f}% |\n")

    # Combined Top Co-occurrences
    summary.append("\n## Combined Top Co-occurrences (Both Formulas)\n")
    summary.append("| Dimension 1 | Dimension 2 | Count |\n")
    summary.append("|------------|-------------|-------|\n")

    # Sort by count
    sorted_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)
    for (dim1, dim2), count in sorted_pairs[:20]:  # Top 20
        summary.append(f"| {dim1} | {dim2} | {count} |\n")

    # Label Distribution by Formula
    summary.append("\n## Label Distribution by Formula\n")

    summary.append("### Formula 4\n")
    f4_label_dist = defaultdict(int)
    for result in f4_labeled:
        n_labels = len(result['labels'])
        f4_label_dist[n_labels] += 1

    for n in sorted(f4_label_dist.keys()):
        summary.append(f"- {n} label(s): {f4_label_dist[n]} arguments\n")

    summary.append("### Formula 5\n")
    f5_label_dist = defaultdict(int)
    for result in f5_labeled:
        n_labels = len(result['labels'])
        f5_label_dist[n_labels] += 1

    for n in sorted(f5_label_dist.keys()):
        summary.append(f"- {n} label(s): {f5_label_dist[n]} arguments\n")

    # Write summary
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(summary)

    print(f"Statistics summary saved to: {output_path}")


def main():
    """Main execution"""
    print("=" * 80)
    print("Anti-Arguments Dimension Labeling")
    print("=" * 80)

    # Initialize API client
    if not API_KEY or not API_URL:
        print("Error: API credentials not found")
        return

    try:
        client = make_api_client(API_KEY, API_URL)
        print(f"API client ready: {LABEL_MODEL}")
    except Exception as e:
        print(f"Error: Failed to initialize API client: {e}")
        return

    # Load filtered arguments
    f4_file = DATA_DIR / "filtered_anti_formula_4_from_raw.json"
    f5_file = DATA_DIR / "filtered_anti_formula_5_from_raw.json"

    if not f4_file.exists():
        print(f"Error: {f4_file} not found")
        return
    if not f5_file.exists():
        print(f"Error: {f5_file} not found")
        return

    with open(f4_file, 'r', encoding='utf-8') as f:
        f4_data = json.load(f)
    with open(f5_file, 'r', encoding='utf-8') as f:
        f5_data = json.load(f)

    print(f"\nLoaded data:")
    print(f"  Formula 4: {sum(len(args) for args in f4_data.values())} arguments")
    print(f"  Formula 5: {sum(len(args) for args in f5_data.values())} arguments")

    # Label arguments
    f4_labeled = label_arguments_batch(client, f4_data, 4)
    f5_labeled = label_arguments_batch(client, f5_data, 5)

    all_labeled = f4_labeled + f5_labeled

    # Build SEPARATE co-occurrence matrices for each formula
    print("\nBuilding co-occurrence matrices...")
    f4_cooc, f4_counts = build_co_occurrence_matrix(f4_labeled)
    f5_cooc, f5_counts = build_co_occurrence_matrix(f5_labeled)
    all_cooc, all_counts = build_co_occurrence_matrix(all_labeled)

    # Create formula-specific network graphs
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    f4_graph_path = OUTPUT_DIR / "dimension_network_formula_4.png"
    f5_graph_path = OUTPUT_DIR / "dimension_network_formula_5.png"

    create_network_graph(f4_cooc, f4_counts, 4, f4_graph_path)
    create_network_graph(f5_cooc, f5_counts, 5, f5_graph_path)

    # Create statistics summary
    summary_path = OUTPUT_DIR / "labeling_statistics.md"
    create_statistics_summary(f4_labeled, f5_labeled, all_counts, all_cooc, summary_path)

    # Save labeled data (labels list + multi-hot vector per argument)
    f4_with_vectors = [{**r, 'vector': labels_to_vector(r['labels'])} for r in f4_labeled]
    f5_with_vectors = [{**r, 'vector': labels_to_vector(r['labels'])} for r in f5_labeled]

    labeled_data_path = OUTPUT_DIR / "labeled_arguments.json"
    with open(labeled_data_path, 'w', encoding='utf-8') as f:
        json.dump({
            'formula_4': f4_with_vectors,
            'formula_5': f5_with_vectors
        }, f, indent=2, ensure_ascii=False)
    print(f"Labeled data saved to: {labeled_data_path}")

    # Save standalone multi-hot tables (JSON + CSV)
    save_multihot_tables(f4_labeled, f5_labeled, OUTPUT_DIR)

    # Save co-occurrence matrices (formula-specific + combined)
    cooc_path = OUTPUT_DIR / "co_occurrence_matrices.json"
    with open(cooc_path, 'w', encoding='utf-8') as f:
        json.dump({
            'formula_4': {
                'co_occurrence': {f"{k[0]}-{k[1]}": v for k, v in f4_cooc.items()},
                'dimension_counts': f4_counts
            },
            'formula_5': {
                'co_occurrence': {f"{k[0]}-{k[1]}": v for k, v in f5_cooc.items()},
                'dimension_counts': f5_counts
            },
            'combined': {
                'co_occurrence': {f"{k[0]}-{k[1]}": v for k, v in all_cooc.items()},
                'dimension_counts': all_counts
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"Co-occurrence matrices saved to: {cooc_path}")

    # Verification
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    # Check 1: All arguments have at least 1 label
    unlabeled = [r for r in all_labeled if len(r['labels']) == 0]
    if unlabeled:
        print(f"[FAIL] {len(unlabeled)} arguments have no labels")
    else:
        print(f"[PASS] All {len(all_labeled)} arguments have at least 1 label")

    # Check 2: All labels are within valid dimensions
    invalid_labels = []
    for r in all_labeled:
        for label in r['labels']:
            if label not in DIMENSIONS:
                invalid_labels.append((r['key'], label))

    if invalid_labels:
        print(f"[FAIL] {len(invalid_labels)} invalid labels found")
        for key, label in invalid_labels[:5]:
            print(f"  {key}: {label}")
    else:
        print("[PASS] All labels are within valid dimensions")

    # Check 3: Both formula-specific network graphs created
    graphs_ok = True
    if f4_graph_path.exists():
        print(f"[PASS] Formula 4 network graph created: {f4_graph_path.name}")
    else:
        print(f"[FAIL] Formula 4 network graph not created")
        graphs_ok = False

    if f5_graph_path.exists():
        print(f"[PASS] Formula 5 network graph created: {f5_graph_path.name}")
    else:
        print(f"[FAIL] Formula 5 network graph not created")
        graphs_ok = False

    if graphs_ok:
        print(f"[PASS] Both formula-specific network graphs created successfully")

    print("\n" + "=" * 80)
    print("SUCCESS: Labeling complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
