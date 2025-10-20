# Update <lib>_apis.txt if you want to run on a subset of APIs

set -euo pipefail

lib=${1:-"torch"}  # Default to "torch" if not provided
llm=${2:-"gemini"}  # Default to "gemini" if not provided

if [ "$lib" != "torch" ] && [ "$lib" != "tf" ]; then
    echo "Usage: $0 [torch|tf] [LLM (Default: gemini)]"
    exit 1
fi

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r llm/requirements.txt

timestamp=$(date -u +"%Y%m%d-%H%M%S-%3N")
llm_dir="llm/$llm"
rulegen_dir="rulegen/$llm"

# Rotate llm directory if it exists, otherwise create fresh
if [ -d "$llm_dir" ]; then
    archived_llm="${llm_dir}-old-${timestamp}"
    mv "$llm_dir" "$archived_llm"
    echo "Archived existing $llm_dir to $archived_llm"
fi
mkdir -p "$llm_dir/inputs"

# Initialize baseline files in llm directory if they are missing
[ -f "$llm_dir/signatures.json" ] || echo "{}" > "$llm_dir/signatures.json"
[ -f "$llm_dir/torch_signatures.py" ] || echo "signatures = {}" > "$llm_dir/torch_signatures.py"
[ -f "$llm_dir/tf_signatures.py" ] || echo "signatures = {}" > "$llm_dir/tf_signatures.py"
[ -f "$llm_dir/valid_inputs_torch.py" ] || echo "generated_inputs = {}" > "$llm_dir/valid_inputs_torch.py"
[ -f "$llm_dir/valid_inputs_tf.py" ] || echo "generated_inputs = {}" > "$llm_dir/valid_inputs_tf.py"
[ -f "$llm_dir/inputs/__init__.py" ] || touch "$llm_dir/inputs/__init__.py"

# Rotate rulegen directory if necessary, then recreate
if [ -d "$rulegen_dir" ]; then
    archived_rulegen="${rulegen_dir}-old-${timestamp}"
    mv "$rulegen_dir" "$archived_rulegen"
    echo "Archived existing $rulegen_dir to $archived_rulegen"
fi
mkdir -p "$rulegen_dir"

# Step 1: Generate signatures and sync APIs with signatures
python -m llm.create_signatures $lib $llm
python -m llm.sync_apis_with_signatures $lib $llm

# Step 2: Generate valid seed inputs
python -m llm.generate_valid_inputs $lib $llm

# Step 3: Run rule generation for each API
cp llm/"$llm"/signature.json .
cd rulegen
python rulegen.py $lib $llm
python translator.py $lib $llm

# Step 4: Only add everything under llm/<llm> and rulegen/<llm> to git
cd ..
git add llm/"$llm"
git add rulegen/"$llm"

echo "Completed LLM pipeline for $lib using $llm."
echo "Please review the changes and commit them."
echo "To run end to end, update the root signatures.json, rules-$lib and llm/valid_inputs_$lib.py files/directories with the llm specific ones."
