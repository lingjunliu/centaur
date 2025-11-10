# Tensorflow

regen=1  # Regenerate invariants flag for all runs
num_p=100  # Maximum number of parallel jobs for Slurm
cp tf_variations.txt tf_variations.txt.bak

# No rule refinement ###################################

source venv/bin/activate
cp tf_ablation_apis.txt tf_apis.txt
python -m utils.sync_apis_and_variations tf
rm -r corpus_tf/*

echo "--------------------------------"
echo "Running pipeline without rule refinement..."
echo "--------------------------------"
bash pipeline.sh tf 1 0 tf_ablation_no_rule_refinement $regen $num_p

#########################################################

# No DocErr #############################################

source venv/bin/activate
cp tf_variations.txt.bak tf_variations.txt
cp tf_ablation_apis.txt tf_apis.txt
python -m utils.sync_apis_and_variations tf
rm -r corpus_tf/*
rm -r rules-tf
mv ablation/rules-tf-no-doc-err-msg rules-tf

if [ $? -ne 0 ]; then
    echo "Error: Failed to move rules-tf-no-doc-err-msg to rules-tf."
    exit 1
fi

echo "--------------------------------"
echo "Running pipeline without DocErr..."
echo "--------------------------------"
bash pipeline.sh tf 1 1 tf_ablation_no_doc_err $regen $num_p

#########################################################

# No Feedback ###########################################

source venv/bin/activate
# cp tf_variations.txt.bak tf_variations.txt
cp tf_ablation_apis.txt tf_apis.txt
python -m utils.sync_apis_and_variations tf
rm -r corpus_tf/*
rm -r rules-tf
mv ablation/rules-tf-no-feedback rules-tf

if [ $? -ne 0 ]; then
    echo "Error: Failed to move rules-tf-no-feedback to rules-tf."
    exit 1
fi

echo "--------------------------------"
echo "Running pipeline without Feedback..."
echo "--------------------------------"
bash pipeline.sh tf 1 1 tf_ablation_no_feedback $regen $num_p

#########################################################

# No Example Rules ######################################

source venv/bin/activate
cp tf_variations.txt.bak tf_variations.txt
cp tf_ablation_apis.txt tf_apis.txt
python -m utils.sync_apis_and_variations tf
rm -r corpus_tf/*
rm -r rules-tf
mv ablation/rules-tf-no-exrules rules-tf

if [ $? -ne 0 ]; then
    echo "Error: Failed to move rules-tf-no-exrules to rules-tf."
    exit 1
fi

echo "--------------------------------"
echo "Running pipeline without Example Rules..."
echo "--------------------------------"
bash pipeline.sh tf 1 1 tf_ablation_no_example_rules $regen $num_p

#########################################################

# Remove everything #######################################

source venv/bin/activate
cp tf_variations.txt.bak tf_variations.txt
cp tf_ablation_apis.txt tf_apis.txt
python -m utils.sync_apis_and_variations tf
rm -r corpus_tf/*
rm -r rules-tf
mv ablation/rules-tf-no-doc-err-msg-feedback-exrules rules-tf

if [ $? -ne 0 ]; then
    echo "Error: Failed to move rules-tf-no-doc-err-msg-feedback-exrules to rules-tf."
    exit 1
fi

echo "--------------------------------"
echo "Running pipeline without all features..."
echo "--------------------------------"
bash pipeline.sh tf 1 0 tf_ablation_no_all_features $regen $num_p

#########################################################