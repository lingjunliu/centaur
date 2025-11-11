#!/bin/bash

run_llm() {
    lib=$1
    llm=$2
    regen=1

    # Inputs and signatures
    cp llm/${llm}-${lib}/valid_inputs_${lib}.py llm/
    cp llm/${llm}-${lib}/signatures.json .

    # APIS
    cp ${lib}_ablation_apis.txt ${lib}_apis.txt
    source venv/bin/activate
    python -m utils.sync_apis_and_variations ${lib}
    # Set num_p to the number of lines in the variations file
    num_p=$(wc -l < ${lib}_variations.txt)
    echo "--------------------------------"
    echo "${llm}: Number of parallel jobs for Slurm: ${num_p}"
    echo "--------------------------------"    

    # Rules
    rm -r rules-${lib}
    cp -r rulegen/${llm}-${lib}/rules-${lib} .

    # Clear corpus
    rm -r corpus_${lib}/*

    echo "--------------------------------"
    echo "Running pipeline for ${llm} on ${lib}..."
    echo "--------------------------------"    
    echo "bash pipeline.sh ${lib} 1 1 ${lib}_${llm} $regen $num_p"
    bash pipeline.sh ${lib} 1 1 ${lib}_${llm} $regen $num_p
}

# Torch, Claude
lib=torch
llm=claude
run_llm $lib $llm

