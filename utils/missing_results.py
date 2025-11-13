import sys

def main():
    lib = sys.argv[1]
    dir = sys.argv[2] if len(sys.argv) > 2 else ".tmp"

    api_list_file = f"{lib}_apis.txt"
    variations_file = f"{lib}_variations.txt"

    with open(api_list_file, 'r') as f:
        apis = [line.strip() for line in f.readlines()]

    with open(variations_file, 'r') as f:
        variations = [line.strip() for line in f.readlines()]

    invariants_file = f"{dir}/infer_result_{lib}.csv"
    models_file = f"{dir}/model_generation_{lib}.csv"
    fuzz_file = f"{dir}/fuzz_result_{lib}.csv"
    cov_file = f"{dir}/coverage_{lib}.csv"

    with open(invariants_file, 'r') as f:
        invariant_apis = [line.strip().split(",")[0] for line in f.readlines()]
        invariant_apis = invariant_apis[1:]  # Skip header
    missing_invariants = set(apis) - set(invariant_apis)
    with open(f"{dir}/missing_invariants_{lib}.txt", 'w') as f:
        for api in sorted(missing_invariants):
            f.write(f"{api}\n")
    print(f"Missing invariants: {len(missing_invariants)}")
    
    with open(models_file, 'r') as f:
        model_apis = [line.strip().split(",")[0] for line in f.readlines()]
        model_apis = model_apis[1:]  # Skip header
    missing_models = set(variations) - set(model_apis)
    with open(f"{dir}/missing_models_{lib}.txt", 'w') as f:
        for api in sorted(missing_models):
            f.write(f"{api}\n")
    print(f"Missing models: {len(missing_models)}")
    
    with open(fuzz_file, 'r') as f:
        fuzzed_apis = []
        for line in f.readlines()[1:]:  # Skip header
            parts = line.strip().split(",")
            api_name = parts[0]
            total = int(parts[-2])
            if total > 0:
                fuzzed_apis.append(api_name)
    missing_fuzz = set(apis) - set(fuzzed_apis)
    with open(f"{dir}/missing_fuzz_{lib}.txt", 'w') as f:
        for api in sorted(missing_fuzz):
            f.write(f"{api}\n")
    print(f"Missing fuzz: {len(missing_fuzz)}")
    
    with open(cov_file, 'r') as f:
        covered_apis = [line.strip().split(",")[0] for line in f.readlines()]
        covered_apis = covered_apis[1:]  # Skip header
    missing_coverage = set(fuzzed_apis) - set(covered_apis)
    with open(f"{dir}/missing_coverage_{lib}.txt", 'w') as f:
        for api in sorted(missing_coverage):
            f.write(f"{api}\n")
    print(f"Missing coverage (fuzzed but not covered): {len(missing_coverage)}")

if __name__ == "__main__":
    main()