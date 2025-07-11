# Tests the pipeline on one APIL: torch.nn.L1Loss (Change it to run for something else)
# Designed to take ~10 minutes

api=torch.nn.L1Loss
lib=torch
variation=torch.nn.L1Loss   # Some APIs will have more

seed=200

# Setup env
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the steps

# Step 1: Infer invariants: <variation> <duration> <regen> <library>
python -m learner.invariant_inference $variation 100 1 || { echo "Inferring failed. "; exit 1; }
# Step 2: Generate models: <variation> <duration> <n_models> <library> <seed> <regen>
python -m generator.z3 $variation 300 0 $lib $seed 1 || { echo "Model gen failed. "; exit 1; }
# Step 3: Fuzz with the generated models: <api> <duration> <n_inputs> <library> <seed>
python -m generator.harness_z3 $api 60 0 $lib $seed || { echo "Fuzzing failed. "; exit 1; }
# Step 4.1: Patch: <api> <n_inputs>
pip install torch==2.2.0
python -m eval.patching $api -1 || { echo "Patching failed. "; exit 1; }
pip install instrumented_pytorch/torch-2.2.0*
# Step 4.2: Coverage: <api> <format>
python -m eval.coverage $api "html" || { echo "Coverage collection failed. "; exit 1; }
pip install torch==2.2.0