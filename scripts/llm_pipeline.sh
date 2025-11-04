# Update <lib>_apis.txt if you want to run on a subset of APIs

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

# Check if llm/<llm> directory exists, if not create it
mkdir -p llm/"$llm"

# Check if llm/<llm>/inputs directory exists, if not create it
mkdir -p llm/"$llm"/inputs

# Check if llm/<llm>/signatures.json file exists, if not create it as an empty json
if [ ! -f llm/"$llm"/signatures.json ]; then
    echo "{}" > llm/"$llm"/signatures.json
fi

# Check if llm/<llm>/<lib>_signatures.py file exists, if not create it with signatures = {}
if [ ! -f llm/"$llm/torch_signatures.py" ]; then
    echo "signatures = {}" > llm/"$llm/torch_signatures.py"
fi

if [ ! -f llm/"$llm/tf_signatures.py" ]; then
    echo "signatures = {}" > llm/"$llm/tf_signatures.py"
fi

# Check if llm/<llm>/valid_inputs_<lib>.py file exists, if not create it with generated_inputs = {}
if [ ! -f llm/"$llm/valid_inputs_torch.py" ]; then
    echo "generated_inputs = {}" > llm/"$llm/valid_inputs_torch.py"
fi

if [ ! -f llm/"$llm/valid_inputs_tf.py" ]; then
    echo "generated_inputs = {}" > llm/"$llm/valid_inputs_tf.py"
fi

# Step 1: Generate signatures and sync APIs with signatures
python -m llm.create_signatures $lib $llm
python -m llm.sync_apis_with_signatures $lib $llm

# Step 2: Generate valid seed inputs
python -m llm.generate_valid_inputs $lib $llm

# Step 3: Run rule generation for each API
cp llm/"$llm"/signatures.json .
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
