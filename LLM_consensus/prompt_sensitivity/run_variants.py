# Run the prompt-sensitivity experiment: 3 variants x 4 models x 5 runs at temp 0.0.
# Appends raw responses to results/raw/<variant>/<model>.md using the same run-header
# format as the root pipeline so the existing regex extractor can parse them.
# Resume-safe: completed runs are detected and skipped.

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIStatusError, APITimeoutError, NotFoundError, OpenAI

ENCODING = "utf-8"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
load_dotenv(REPO_ROOT / ".env")

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

RUNS = 3  # minimal viable: modal winner needs n>=3; rho/shifts use mean tables
TEMPERATURE = 0.0
SLEEP_SECONDS = 3

MODELS = [
    "claude-opus-4-6",
    "grok-4-5",
]

VARIANTS = {
    "baseline_prompt": REPO_ROOT / "prompt.md",
    "v1_reorder": HERE / "variants" / "prompt_v1_reorder.md",
    "v2_paraphrase": HERE / "variants" / "prompt_v2_paraphrase.md",
    "v3_no_optimization": HERE / "variants" / "prompt_v3_no_optimization.md",
}

RAW_DIR = HERE / "results" / "raw"

RUN_HEADER = "# Run {i} response, Temperature =0.0:"
RUN_HEADER_RE = re.compile(r"# Run (\d+) response")


def completed_runs(path: Path) -> set:
    """Return run indices already present in the raw output file."""
    if not path.exists():
        return set()
    text = path.read_text(encoding=ENCODING)
    return {int(m) for m in RUN_HEADER_RE.findall(text)}


def get_response(client: OpenAI, model: str, message: str) -> str:
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            stream=False,
            temperature=TEMPERATURE,
        )
        return stream.choices[0].message.content or ""
    except NotFoundError:
        print(f"[Error] {model}: model not found")
        return ""
    except APITimeoutError:
        print(f"[Error] {model}: API timeout")
        return ""
    except APIStatusError as err:
        try:
            detail = err.response.json().get("error", {}).get("message", "unknown error")
        except Exception:
            detail = "unknown error"
        print(f"[Error] {model}: API status error: {detail}")
        return ""
    except Exception as err:
        print(f"[Error] {model}: unexpected: {type(err).__name__}: {err}")
        return ""


def main() -> None:
    if not (API_KEY and API_URL):
        print("[Fail] API_KEY / API_URL missing; check .env at repo root")
        sys.exit(1)

    prompts = {}
    for name, path in VARIANTS.items():
        if not path.exists():
            print(f"[Fail] variant prompt missing: {path}")
            sys.exit(1)
        prompts[name] = path.read_text(encoding=ENCODING)
    print(f"[Okay] loaded {len(prompts)} variant prompts")

    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    print("[Okay] client created")

    pending = []
    for vname in VARIANTS:
        for model in MODELS:
            out_path = RAW_DIR / vname / f"{model}.md"
            done = completed_runs(out_path)
            todo = [i for i in range(RUNS) if i not in done]
            pending.append((vname, model, out_path, todo))

    total_todo = sum(len(todo) for _, _, _, todo in pending)
    print(f"[Info] {total_todo} calls pending (of {len(VARIANTS) * len(MODELS) * RUNS})")
    if total_todo == 0:
        print("[Okay] nothing to do")
        return

    failures = []
    done_count = 0
    for vname, model, out_path, todo in pending:
        if not todo:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for run_i in todo:
            done_count += 1
            print(f"[Run {done_count}/{total_todo}] variant={vname} model={model} run={run_i}")
            response = get_response(client, model, prompts[vname])
            if not response.strip():
                failures.append((vname, model, run_i))
                response = "[EXTRACTION FAILED: empty response]"
            with open(out_path, "a", encoding=ENCODING) as fh:
                fh.write(RUN_HEADER.format(i=run_i) + "\n\n")
                fh.write(response)
                fh.write("\n\n")
            time.sleep(SLEEP_SECONDS)

    print(f"[Okay] finished; {len(failures)} failed calls")
    for vname, model, run_i in failures:
        print(f"  failed: {vname} {model} run {run_i}")


if __name__ == "__main__":
    main()
