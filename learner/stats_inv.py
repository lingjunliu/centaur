import os, sys
import numpy as np
from utils.misc import get_dir_in_root, get_tmp_dir, read_file_in_root, bcolors
from utils.new_api_utils import get_api_suffix
from llm.valid_inputs_torch import generated_inputs as generated_inputs_torch
from llm.valid_inputs_tf import generated_inputs as generated_inputs_tf

def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"

    # alias
    if lib == "pytorch":
        lib = "torch"
    elif lib == "tensorflow":
        lib = "tf"
        
    supported_apis = read_file_in_root(f"{lib}_apis.txt")
    supported_variations = read_file_in_root(f"{lib}_variations.txt")
    generated_inputs = generated_inputs_torch if lib == "torch" else generated_inputs_tf
    outdated_apis = set()

    rule_to_api = {}
    api_to_rule = {}
    inv_dir = get_dir_in_root(f"invariants_{lib}")
    for file in os.listdir(inv_dir):
        if file not in supported_variations:
            print(f"Invariants learned for unsupported api variation {file}")
            outdated_apis.add(file)
            continue
        file_path = os.path.join(inv_dir, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                for line in f.readlines():
                    tokens = line.strip().split(",")
                    api, rule = tokens[0], tokens[2]
                    
                    # rule to api mapping
                    if rule not in rule_to_api:
                        rule_to_api[rule] = set()
                    rule_to_api[rule].add(api)
                    
                    # api to rule mapping
                    if api not in api_to_rule:
                        api_to_rule[api] = set()
                    api_to_rule[api].add(rule)

    apis_with_invariants = set(api_to_rule.keys())
    apis_without_invariants = set(supported_apis) - apis_with_invariants
    
    corpus_folder = get_dir_in_root(f"corpus_{lib}")

    apis_with_models = set()
    for variation in supported_variations:
        model_path = os.path.join(corpus_folder, variation)
        if os.path.isdir(model_path):
            api, suffix = get_api_suffix(variation)
            if api in supported_apis:
                n_models = len(os.listdir(model_path))
                if n_models > 0:
                    apis_with_models.add(api)
    
    apis_without_models = set(supported_apis) - apis_with_models

    variations_with_llm_inputs = set(generated_inputs.keys())
    apis_with_llm_inputs = set()

    for variation in variations_with_llm_inputs:
        api, suffix = get_api_suffix(variation)
        if api in supported_apis:
            apis_with_llm_inputs.add(api)
    
    apis_without_llm_inputs = set(supported_apis) - apis_with_llm_inputs

    apis_without_invariants_but_with_models = apis_without_invariants.intersection(apis_with_models)
    if len(apis_without_invariants_but_with_models) > 0:
        print(f"\n{bcolors.WARNING}WARNING: There are {len(apis_without_invariants_but_with_models)} APIs without invariants but they have models!{bcolors.ENDC}")


    apis_with_models = apis_with_models.intersection(apis_with_invariants)

    stats_str = "Library,Target APIs,Has LLM Generated Inputs,Has Invariants,Has Models\n"
    stats_str += f"{lib},{len(supported_apis)},{len(apis_with_llm_inputs)},{len(apis_with_invariants)},{len(apis_with_models)}\n"

    rng = np.random.default_rng(21)

    sample_apis_without_inputs = rng.choice(list(apis_without_llm_inputs), size=min(5, len(apis_without_llm_inputs)), replace=False)
    sample_apis_without_invariants = rng.choice(list(apis_without_invariants.intersection(apis_with_llm_inputs)), size=min(5, len(apis_without_invariants.intersection(apis_without_invariants.intersection(apis_with_llm_inputs)))), replace=False)
    sample_apis_without_models = rng.choice(list(apis_without_models.intersection(apis_with_invariants)), size=min(5, len(apis_without_models.intersection(apis_with_invariants))), replace=False)

    print("\nSample APIs without LLM generated inputs:\n")
    print('\n'.join(sample_apis_without_inputs))
    print("\nSample APIs without invariants:\n")
    print('\n'.join(sample_apis_without_invariants))
    print("\nSample APIs without models:\n")
    print('\n'.join(sample_apis_without_models))

    # Save stats to file
    tmp = get_tmp_dir()
    rule_to_api_csv = os.path.join(tmp, f"rule_to_api_{lib}.csv")
    api_to_rule_csv = os.path.join(tmp, f"api_to_rule_{lib}.csv")
    stats_csv = os.path.join(tmp, f"stats_{lib}.csv")
    outdated_file = os.path.join(tmp, f"outdated_apis_{lib}.txt")
    models_without_invariants = os.path.join(tmp, f"models_without_invariants_{lib}.txt")
    apis_without_invariants_file = os.path.join(tmp, f"apis_without_invariants_{lib}.txt")
    
    with open(rule_to_api_csv, "w") as f:
        for rule, apis in rule_to_api.items():
            f.write(f"{rule},{len(apis)}\n")
            
    with open(api_to_rule_csv, "w") as f:
        for api, rules in api_to_rule.items():
            f.write(f"{api},{len(rules)}\n")

    with open(stats_csv, "w") as f:
        f.write(stats_str)

    with open(outdated_file, "w") as f:
        f.write('\n'.join(outdated_apis))

    with open(models_without_invariants, "w") as f:
        f.write('\n'.join(sorted(apis_without_invariants_but_with_models)))

    with open(apis_without_invariants_file, "w") as f:
        f.write('\n'.join(sorted(apis_without_invariants)))
if __name__ == "__main__":
    main()