import pandas as pd
import sys, os

def main():
    dir_1 = sys.argv[1]
    dir_2 = sys.argv[2]
    suffix_1 = sys.argv[3] if len(sys.argv) > 3 else "one"
    suffix_2 = sys.argv[4] if len(sys.argv) > 4 else "two"
    lib = sys.argv[5] if len(sys.argv) > 5 else "torch"
    output_dir = sys.argv[6] if len(sys.argv) > 6 else ".tmp"

    os.makedirs(output_dir, exist_ok=True)

    cov_filename = f"coverage_{lib}.csv"
    fuzz_filename = f"fuzz_result_{lib}.csv"

    cov_1 = pd.read_csv(os.path.join(dir_1, cov_filename))
    cov_2 = pd.read_csv(os.path.join(dir_2, cov_filename))

    # Only keep the columns 'api' and 'SLATE'
    cov_1 = cov_1[['api', 'SLATE']]
    cov_2 = cov_2[['api', 'SLATE']]

    # Rename the 'SLATE' column to 'cov'
    cov_1 = cov_1.rename(columns={'SLATE': 'cov'})
    cov_2 = cov_2.rename(columns={'SLATE': 'cov'})

    fuzz_1 = pd.read_csv(os.path.join(dir_1, fuzz_filename))
    fuzz_2 = pd.read_csv(os.path.join(dir_2, fuzz_filename))

    # Only keep the columns 'api', 'n_models', 'total', 'valid_prcnt'
    fuzz_1 = fuzz_1[['api', 'n_models', 'total', 'valid_prcnt']]
    fuzz_2 = fuzz_2[['api', 'n_models', 'total', 'valid_prcnt']]

    # Get rows with matching value in the 'api' column
    merged_cov = pd.merge(cov_1, cov_2, on='api', suffixes=(f"_{suffix_1}", f"_{suffix_2}"))
    merged_fuzz = pd.merge(fuzz_1, fuzz_2, on='api', suffixes=(f"_{suffix_1}", f"_{suffix_2}"))

    # Get a list of apis that are in cov_1 but not in cov_2 and vice versa
    apis_only_in_1 = set(cov_1['api']) - set(cov_2['api'])
    apis_only_in_2 = set(cov_2['api']) - set(cov_1['api'])

    # Save results
    merged_cov.to_csv(os.path.join(output_dir, f"{suffix_1}_vs_{suffix_2}_cov_{lib}.csv"), index=False)
    merged_fuzz.to_csv(os.path.join(output_dir, f"{suffix_1}_vs_{suffix_2}_val_{lib}.csv"), index=False)
    
    with open(os.path.join(output_dir, f"{suffix_1}_vs_{suffix_2}_apis_{lib}.txt"), 'w') as f:
        if len(apis_only_in_1) == 0 and len(apis_only_in_2) == 0:
            f.write("All APIs match!\n")
        elif len(apis_only_in_1) == 0:
            f.write("All APIs in " + suffix_1 + " are also in " + suffix_2 + "\n\n")
        elif len(apis_only_in_2) == 0:
            f.write("All APIs in " + suffix_2 + " are also in " + suffix_1 + "\n\n")

        if len(apis_only_in_1) > 0:
            f.write("Only in " + suffix_1 + ":\n")
            for api in apis_only_in_1:
                f.write(api + "\n")
        if len(apis_only_in_2) > 0:
            f.write("\nOnly in " + suffix_2 + ":\n")
            for api in apis_only_in_2:
                f.write(api + "\n")

if __name__ == "__main__":
    main()