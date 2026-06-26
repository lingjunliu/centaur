#!/usr/bin/env bash
# Standard rule-regeneration pipeline for one library (tf or torch).
#
# Steps:
#   1. Run the translator (regenerates rules-<lib>/ from rules-ebnf).
#   2. Classify the diff and revert reorder-only files (set-iteration noise).
#   3. Build the list of APIs that have *substantive* changes.
#   4. Write that API list to llm/gemini/<lib>_variations.txt.
#   5. Generate valid inputs:  python -m llm.generate_valid_inputs <lib> gemini
#   6. Copy llm/gemini/valid_inputs_<lib>.py -> llm/valid_inputs_<lib>.py
#   7. Write the same API list to the root <lib>_variations.txt.
#   8. (printed only) Infer invariants via SLURM.
#
# Usage: bash scripts/regen_pipeline.sh <tf|torch>
set -euo pipefail

lib="${1:-}"
if [[ "$lib" != "tf" && "$lib" != "torch" ]]; then
    echo "Usage: $0 <tf|torch>" >&2
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
echo "==> [1/7] Running translator for ${lib}..."
( cd rulegen && python3 translator.py "$lib" )

# ---------------------------------------------------------------------------
echo "==> [2/7] Classifying diff and reverting reorder-only files..."
python3 classify.py "$lib"
reorder_file="reorder_only_${lib}.txt"
if [[ -s "$reorder_file" ]]; then
    count=$(grep -c . "$reorder_file")
    echo "    Reverting ${count} reorder-only files..."
    tr '\n' '\0' < "$reorder_file" | xargs -0 --no-run-if-empty git checkout --
else
    echo "    No reorder-only files to revert."
fi

# ---------------------------------------------------------------------------
echo "==> [3/7] Building substantive API list..."
substantive_file="substantive_changes_${lib}.txt"
api_list="substantive_apis_${lib}.txt"
if [[ -s "$substantive_file" ]]; then
    sed "s#rules-${lib}/##; s#/[^/]*\$##" "$substantive_file" | sort -u > "$api_list"
else
    : > "$api_list"
fi
napis=$(grep -c . "$api_list" || true)
echo "    ${napis} APIs with substantive changes -> ${api_list}"

# ---------------------------------------------------------------------------
echo "==> [4/7] Updating llm/gemini/${lib}_variations.txt..."
cp "$api_list" "llm/gemini/${lib}_variations.txt"

# ---------------------------------------------------------------------------
echo "==> [5/7] Generating valid inputs (python -m llm.generate_valid_inputs ${lib} gemini)..."
python -m llm.generate_valid_inputs "$lib" gemini

# ---------------------------------------------------------------------------
echo "==> [6/7] Copying llm/gemini/valid_inputs_${lib}.py -> llm/valid_inputs_${lib}.py..."
cp "llm/gemini/valid_inputs_${lib}.py" "llm/valid_inputs_${lib}.py"

# ---------------------------------------------------------------------------
echo "==> [7/7] Updating root ${lib}_variations.txt..."
cp "$api_list" "${lib}_variations.txt"

# ---------------------------------------------------------------------------
echo
echo "==> Steps 1-7 complete for ${lib}."
echo "    Step 8 (infer invariants via SLURM) — run manually when ready:"
echo
echo "        bash scripts/infer_invariants_with_slurm.sh 300 1 ${lib} 1 42"
echo
