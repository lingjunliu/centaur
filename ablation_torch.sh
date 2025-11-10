# Torch

regen=1  # Regenerate invariants flag for all runs
num_p=141  # Maximum number of parallel jobs for Slurm
cp torch_variations.txt torch_variations.txt.bak

# No rule refinement ###################################

source venv/bin/activate
cp torch_ablation_apis.txt torch_apis.txt
python -m utils.sync_apis_and_variations torch
rm -r corpus_torch/*

echo "--------------------------------"
echo "Running pipeline without rule refinement..."
echo "--------------------------------"
bash pipeline.sh torch 1 0 torch_ablation_no_rule_refinement

#########################################################

# No DocErr #############################################

source venv/bin/activate
cp torch_variations.txt.bak torch_variations.txt
cp torch_ablation_apis.txt torch_apis.txt
python -m utils.sync_apis_and_variations torch
rm -r corpus_torch/*
rm -r rules-torch
mv ablation/rules-torch-no-doc-err-msg rules-torch

if [ $? -ne 0 ]; then
    echo "Error: Failed to move rules-torch-no-doc-err-msg to rules-torch."
    exit 1
fi

echo "--------------------------------"
echo "Running pipeline without DocErr..."
echo "--------------------------------"
bash pipeline.sh torch 1 1 torch_ablation_no_doc_err

#########################################################

# No Feedback ###########################################

source venv/bin/activate
# cp torch_variations.txt.bak torch_variations.txt
cp torch_ablation_apis.txt torch_apis.txt
python -m utils.sync_apis_and_variations torch
rm -r corpus_torch/*
rm -r rules-torch
mv ablation/rules-torch-no-feedback rules-torch

if [ $? -ne 0 ]; then
    echo "Error: Failed to move rules-torch-no-feedback to rules-torch."
    exit 1
fi

echo "--------------------------------"
echo "Running pipeline without Feedback..."
echo "--------------------------------"
bash pipeline.sh torch 1 1 torch_ablation_no_feedback

#########################################################

# No Example Rules ######################################

source venv/bin/activate
cp torch_variations.txt.bak torch_variations.txt
cp torch_ablation_apis.txt torch_apis.txt
python -m utils.sync_apis_and_variations torch
rm -r corpus_torch/*
rm -r rules-torch
mv ablation/rules-torch-no-exrules rules-torch

if [ $? -ne 0 ]; then
    echo "Error: Failed to move rules-torch-no-exrules to rules-torch."
    exit 1
fi

echo "--------------------------------"
echo "Running pipeline without Example Rules..."
echo "--------------------------------"
bash pipeline.sh torch 1 1 torch_ablation_no_example_rules

#########################################################

# Remove everything #######################################

source venv/bin/activate
cp torch_variations.txt.bak torch_variations.txt
cp torch_ablation_apis.txt torch_apis.txt
python -m utils.sync_apis_and_variations torch
rm -r corpus_torch/*
rm -r rules-torch
mv ablation/rules-torch-no-doc-err-msg-feedback-exrules rules-torch

if [ $? -ne 0 ]; then
    echo "Error: Failed to move rules-torch-no-doc-err-msg-feedback-exrules to rules-torch."
    exit 1
fi

echo "--------------------------------"
echo "Running pipeline without all features..."
echo "--------------------------------"
bash pipeline.sh torch 1 0 torch_ablation_no_all_features

#########################################################