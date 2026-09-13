# Build prompt variants for the sensitivity experiment.
# V1: candidate list order reversed (benign, position perturbation)
# V2: paraphrased scale anchors / parameter wording (benign, wording perturbation)
# V3: constrained-optimization paradigm ablated (structural perturbation)
# Base text is read from the repo-root prompt.md so no manual transcription occurs.

import sys
from pathlib import Path

ENCODING = "utf-8"

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PROMPT = REPO_ROOT / "prompt.md"
OUT_DIR = Path(__file__).resolve().parent / "variants"


def build_v1_reorder(text: str) -> str:
    """Reverse the order of the 10 formula lines inside the INPUT DATA block."""
    lines = text.splitlines()
    formula_idx = [i for i, ln in enumerate(lines) if ln.strip().startswith("- formula ")]
    if len(formula_idx) != 10:
        raise SystemExit(f"[Fail] expected 10 formula lines, found {len(formula_idx)}")
    block = [lines[i] for i in formula_idx]
    reordered = list(reversed(block))
    for pos, i in enumerate(formula_idx):
        lines[i] = reordered[pos]
    return "\n".join(lines) + "\n"


# (old, new) wording replacement pairs for V2. Each old string must match exactly once.
V2_REPLACEMENTS = [
    (
        """*   **0-3:** Unacceptable / Fatal Flaw (e.g., causes thrombosis, severe inflammation, or structural failure).
*   **4-6:** Marginal / Requires mitigation (e.g., slight mismatch, neutral biological response).
*   **7-9:** Highly Suitable (e.g., matches artery modulus, promotes endothelialization).
*   **10:** Theoretical Optimum.""",
        """*   **0-3:** Disqualified owing to a critical deficiency (for instance, provoking clot formation, pronounced inflammation, or mechanical breakdown).
*   **4-6:** Borderline; corrective measures would be needed (for instance, minor deviation from the target, or an indifferent biological response).
*   **7-9:** Strongly appropriate (for instance, modulus aligned with the arterial wall, supportive of endothelial growth).
*   **10:** Ideal theoretical performance.""",
    ),
    (
        """1. Mechanical Safety (reward materials with Young's Module between the reference modulus range)
2. Swelling Performance (reward high swelling ratio. Severely penalize this parameter if the swelling is uncontrolled or lacks mechanical stability in swollen state.
3. Endothelialization (penalize endothelial toxicity and over-promotion of endothelialization which may lead to occlusion)
4. SMC-inhibition (reward inhibition of smooth muscle proliferation and transdifferation into synthetic phenotype)
5. Anti-inflammation (reward anti-inflammation capacities, consider effects upon both cellular and cytokine level; also penalize hypersensitivity reactions)
6. Thrombogenicity (reward anti-thrombogenic properties)""",
        """1. Mechanical Safety (favor materials whose Young's modulus lies within the referenced arterial range)
2. Swelling Performance (favor a high swelling ratio; assign a severe penalty when swelling is uncontrolled or the swollen state lacks mechanical stability.)
3. Endothelialization (deduct for endothelial toxicity as well as for excessive endothelial growth that could occlude the lumen)
4. SMC-inhibition (favor suppression of smooth muscle proliferation and of its shift toward the synthetic phenotype)
5. Anti-inflammation (favor anti-inflammatory action considered at both the cellular and cytokine levels; additionally deduct for hypersensitivity responses)
6. Thrombogenicity (favor materials that resist thrombus formation)""",
    ),
    (
        "Calculate the Total Score for each candidate. The candidate with the highest Total Score AND no single score below 4 is the WINNER.",
        "Compute the Total Score for every candidate. The candidate achieving the greatest Total Score, provided none of its individual scores fall below 4, is declared the WINNER.",
    ),
    (
        """* Note: Default to 5 if the material lacks sufficient information upon this criteria, and explicitly note \\"Lack of information\\" in your internal logic.""",
        """* Note: Assign a default value of 5 whenever the available information on a criterion is insufficient, and explicitly state \\"Lack of information\\" within your internal reasoning.""",
    ),
]

V3_START_MARKER = "**THE ASSESSMENT PARADIGM: CONSTRAINED OPTIMIZATION**"
V3_END_MARKER = "Use an absolute 0-10 scale defined as:"

V3_NEW = """**ASSESSMENT BASIS**
Evaluate every candidate in its conventional, as-typically-prepared form as reported in the standard literature. Do NOT assume any formulation-specific optimization. Note the fundamental Polymer Physics Trade-offs: 1. *Mechanical Safety* and *Swelling Performance* are coupled mechanistically, so improving one generally affects the other. 2. *Endothelialization*, *Thrombogenicity*, *SMC-inhibition* and *Anti-inflammation* are largely dictated by the inherent chemical properties of the material; base your evaluation on these intrinsic motifs and functional groups.

"""


def build_v2_paraphrase(text: str) -> str:
    for old, new in V2_REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"[Fail] V2 anchor matched {count} times (expected 1): {old[:60]!r}")
        text = text.replace(old, new)
    return text


def build_v3_no_optimization(text: str) -> str:
    # Marker-based splice: replace everything between the two markers, immune to
    # trailing-whitespace drift inside the ablated block.
    start = text.find(V3_START_MARKER)
    end = text.find(V3_END_MARKER)
    if start == -1 or end == -1 or not (start < end):
        raise SystemExit("[Fail] V3 markers not found or out of order")
    return text[:start] + V3_NEW + text[end:]


def main() -> None:
    text = BASE_PROMPT.read_text(encoding=ENCODING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = {
        "v1_reorder": build_v1_reorder(text),
        "v2_paraphrase": build_v2_paraphrase(text),
        "v3_no_optimization": build_v3_no_optimization(text),
    }
    for name, content in variants.items():
        out = OUT_DIR / f"prompt_{name}.md"
        out.write_text(content, encoding=ENCODING)
        print(f"[Okay] wrote {out} ({len(content)} chars)")
    print("[Okay] all variants built")


if __name__ == "__main__":
    sys.exit(main())
