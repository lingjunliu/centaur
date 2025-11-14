import pandas as pd
import sys, os

def main():
    dir_1 = sys.argv[1]
    dir_2 = sys.argv[2]
    suffix_1 = sys.argv[3] if len(sys.argv) > 3 else "one"
    suffix_2 = sys.argv[4] if len(sys.argv) > 4 else "two"
    lib = sys.argv[5] if len(sys.argv) > 5 else "torch"
    output_dir = sys.argv[6] if len(sys.argv) > 6 else ".tmp"

    text_map = {
        "openai": "GPT-5",
        "claude": "Claude Sonnet 4.5",
        "gemma": "Gemma 3 27B",
        "qwen": "Qwen3-Coder-30B"
    }

    target_apis_file = f"{lib}_apis.txt"
    with open(target_apis_file, 'r') as f:
        target_apis = set([line.strip() for line in f.readlines()])

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

    print(f"\\multirow{{3}}{{*}}{{{text_map.get(suffix_2, suffix_2)}}},# APIs,{fuzz_1.shape[0]}/100,{fuzz_2.shape[0]}/100,{'+' if fuzz_1.shape[0] - fuzz_2.shape[0] <=0 else '-'}{abs(fuzz_1.shape[0] - fuzz_2.shape[0])}")

    # Only keep the columns 'api', 'n_models', 'total', 'valid_prcnt'
    fuzz_1 = fuzz_1[['api', 'n_models', 'total', 'valid_prcnt']]
    fuzz_2 = fuzz_2[['api', 'n_models', 'total', 'valid_prcnt']]

    # Get rows with matching value in the 'api' column
    merged_cov = pd.merge(cov_1, cov_2, on='api', suffixes=(f"_{suffix_1}", f"_{suffix_2}"))
    merged_fuzz = pd.merge(fuzz_1, fuzz_2, on='api', suffixes=(f"_{suffix_1}", f"_{suffix_2}"))

    # Add a new column 'diff' to merged_cov which is the difference between the two 'cov' columns
    merged_cov['diff'] = merged_cov[f'cov_{suffix_1}'] - merged_cov[f'cov_{suffix_2}']
    merged_fuzz['diff'] = merged_fuzz[f'valid_prcnt_{suffix_1}'] - merged_fuzz[f'valid_prcnt_{suffix_2}']

    # Add a new row at the end of merged_cov with the average of each column and round to 2 decimal places
    avg_row_cov = pd.DataFrame(merged_cov.mean(numeric_only=True)).T
    avg_row_cov = avg_row_cov.round(2)
    avg_row_cov['api'] = 'average'
    merged_cov = pd.concat([merged_cov, avg_row_cov], ignore_index=True)
    print(f",Coverage (Avg),{avg_row_cov[f'cov_{suffix_1}'].values[0]},{avg_row_cov[f'cov_{suffix_2}'].values[0]},{'+' if avg_row_cov['diff'].values[0] <=0 else '-'}{abs(avg_row_cov['diff'].values[0])}")
    # Add a new row at the end of merged_fuzz with the average of each column
    avg_row_fuzz = pd.DataFrame(merged_fuzz.mean(numeric_only=True)).T
    avg_row_fuzz = avg_row_fuzz.round(2)
    avg_row_fuzz['api'] = 'average'
    merged_fuzz = pd.concat([merged_fuzz, avg_row_fuzz], ignore_index=True)
    print(f",Validity Ratio (Avg),{avg_row_fuzz[f'valid_prcnt_{suffix_1}'].values[0]}%,{avg_row_fuzz[f'valid_prcnt_{suffix_2}'].values[0]}%,{'+' if avg_row_fuzz['diff'].values[0] <=0 else '-'}{abs(avg_row_fuzz['diff'].values[0])}%")
    # Get a list of apis that are in cov_1 but not in cov_2 and vice versa
    apis_only_in_1 = set(cov_1['api']) - set(cov_2['api'])
    apis_only_in_2 = set(cov_2['api']) - set(cov_1['api'])

    apis_only_in_1 = apis_only_in_1.intersection(target_apis)
    apis_only_in_2 = apis_only_in_2.intersection(target_apis)
    missing_either = target_apis - (set(cov_1['api']).union(set(cov_2['api'])))

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

        if len(missing_either) > 0:
            f.write("\nMissing in both:\n")
            for api in missing_either:
                f.write(api + "\n")

if __name__ == "__main__":
    main()