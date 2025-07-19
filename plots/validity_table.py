import pandas as pd

def main():
    titanfuzz_torch = pd.read_csv('data/val_vs_Titanfuzz.csv')
    acetest_torch = pd.read_csv('data/val_vs_ACETest.csv')
    pathfinder_torch = pd.read_csv('data/val_vs_Pathfinder.csv')

    titanfuzz_torch_cov = pd.read_csv('data/vs_Titanfuzz.csv')
    acetest_torch_cov = pd.read_csv('data/vs_ACETest.csv')
    pathfinder_torch_cov = pd.read_csv('data/vs_Pathfinder.csv')

    titanfuzz_tf = pd.read_csv('data/val_vs_Titanfuzz_tf.csv')
    acetest_tf = pd.read_csv('data/val_vs_ACETest_tf.csv')
    pathfinder_tf = pd.read_csv('data/val_vs_Pathfinder_tf.csv')

    titanfuzz_tf_cov = pd.read_csv('data/vs_Titanfuzz_tf.csv')
    acetest_tf_cov = pd.read_csv('data/vs_ACETest_tf.csv')
    pathfinder_tf_cov = pd.read_csv('data/vs_Pathfinder_tf.csv')

    # Titanfuzz
    titanfuzz_total_torch = titanfuzz_torch['total'].sum()
    titanfuzz_valid_torch = titanfuzz_torch['valid'].sum()
    titanfuzz_validity_ratio_torch = (titanfuzz_valid_torch / titanfuzz_total_torch) * 100
    titanfuzz_coverage_torch = titanfuzz_torch_cov['Titanfuzz'].mean()
    print(f"Titanfuzz Torch Validity Ratio: {titanfuzz_validity_ratio_torch:.2f}%")

    slate_titanfuzz_total_torch = titanfuzz_torch['slate_total'].sum()
    slate_titanfuzz_valid_torch = titanfuzz_torch['slate_nominal'].sum()
    slate_titanfuzz_validity_ratio_torch = (slate_titanfuzz_valid_torch / slate_titanfuzz_total_torch) * 100
    slate_titanfuzz_coverage_torch = titanfuzz_torch_cov['SLATE'].mean()
    print(f"Slate Titanfuzz Torch Validity Ratio: {slate_titanfuzz_validity_ratio_torch:.2f}%")

    # Titanfuzz TF
    titanfuzz_total_tf = titanfuzz_tf['total'].sum()
    titanfuzz_valid_tf = titanfuzz_tf['valid'].sum()
    titanfuzz_validity_ratio_tf = (titanfuzz_valid_tf / titanfuzz_total_tf) * 100
    titanfuzz_coverage_tf = titanfuzz_tf_cov['Titanfuzz'].mean()
    print(f"Titanfuzz TF Validity Ratio: {titanfuzz_validity_ratio_tf:.2f}%")

    slate_titanfuzz_total_tf = titanfuzz_tf['slate_total'].sum()
    slate_titanfuzz_valid_tf = titanfuzz_tf['slate_nominal'].sum()
    slate_titanfuzz_validity_ratio_tf = (slate_titanfuzz_valid_tf / slate_titanfuzz_total_tf) * 100
    slate_titanfuzz_coverage_tf = titanfuzz_tf_cov['SLATE'].mean()
    print(f"Slate Titanfuzz TF Validity Ratio: {slate_titanfuzz_validity_ratio_tf:.2f}%")

    # ACETest
    acetest_total_torch = acetest_torch['times'].sum()
    acetest_valid_torch = acetest_torch['times'].sum() - acetest_torch['invalid'].sum()
    acetest_validity_ratio_torch = (acetest_valid_torch / acetest_total_torch) * 100
    acetest_coverage_torch = acetest_torch_cov['ACETest'].mean()
    print(f"ACETest Torch Validity Ratio: {acetest_validity_ratio_torch:.2f}%")

    slate_acetest_total_torch = acetest_torch['slate_total'].sum()
    slate_acetest_valid_torch = acetest_torch['slate_nominal'].sum()
    slate_acetest_validity_ratio_torch = (slate_acetest_valid_torch / slate_acetest_total_torch) * 100
    slate_acetest_coverage_torch = acetest_torch_cov['SLATE'].mean()
    print(f"Slate ACETest Torch Validity Ratio: {slate_acetest_validity_ratio_torch:.2f}%")

    # ACETest TF
    acetest_total_tf = acetest_tf['times'].sum()
    acetest_valid_tf = acetest_tf['times'].sum() - acetest_tf['invalid'].sum()
    acetest_validity_ratio_tf = (acetest_valid_tf / acetest_total_tf) * 100
    acetest_coverage_tf = acetest_tf_cov['ACETest'].mean()
    print(f"ACETest TF Validity Ratio: {acetest_validity_ratio_tf:.2f}%")

    slate_acetest_total_tf = acetest_tf['slate_total'].sum()
    slate_acetest_valid_tf = acetest_tf['slate_nominal'].sum()
    slate_acetest_validity_ratio_tf = (slate_acetest_valid_tf / slate_acetest_total_tf) * 100
    slate_acetest_coverage_tf = acetest_tf_cov['SLATE'].mean()
    print(f"Slate ACETest TF Validity Ratio: {slate_acetest_validity_ratio_tf:.2f}%")

    # Pathfinder
    pathfinder_total_torch = pathfinder_torch['total'].sum()
    pathfinder_valid_torch = pathfinder_torch['valid'].sum()
    pathfinder_validity_ratio_torch = (pathfinder_valid_torch / pathfinder_total_torch) * 100
    pathfinder_coverage_torch = pathfinder_torch_cov['Pathfinder'].mean()
    print(f"Pathfinder Torch Validity Ratio: {pathfinder_validity_ratio_torch:.2f}%")  

    slate_pathfinder_total_torch = pathfinder_torch['slate_total'].sum()
    slate_pathfinder_valid_torch = pathfinder_torch['slate_nominal'].sum()
    slate_pathfinder_validity_ratio_torch = (slate_pathfinder_valid_torch / slate_pathfinder_total_torch) * 100
    slate_pathfinder_coverage_torch = pathfinder_torch_cov['SLATE'].mean()
    print(f"Slate Pathfinder Torch Validity Ratio: {slate_pathfinder_validity_ratio_torch:.2f}%")

    # Pathfinder TF
    pathfinder_total_tf = pathfinder_tf['total'].sum()
    pathfinder_valid_tf = pathfinder_tf['valid'].sum()
    pathfinder_validity_ratio_tf = (pathfinder_valid_tf / pathfinder_total_tf) * 100
    pathfinder_coverage_tf = pathfinder_tf_cov['Pathfinder'].mean()
    print(f"Pathfinder TF Validity Ratio: {pathfinder_validity_ratio_tf:.2f}%")

    slate_pathfinder_total_tf = pathfinder_tf['slate_total'].sum()
    slate_pathfinder_valid_tf = pathfinder_tf['slate_nominal'].sum()
    slate_pathfinder_validity_ratio_tf = (slate_pathfinder_valid_tf / slate_pathfinder_total_tf) * 100
    slate_pathfinder_coverage_tf = pathfinder_tf_cov['SLATE'].mean()
    print(f"Slate Pathfinder TF Validity Ratio: {slate_pathfinder_validity_ratio_tf:.2f}%")

    # Save results to CSV
    header_row = "Tool,\\titanfuzz,\\tname,\\acetest,\\tname,\\pathfinder,\\tname,\\titanfuzz,\\tname,\\acetest,\\tname,\\pathfinder,\\tname\n"
    total_row = f"Total,{int(titanfuzz_total_torch)},{int(slate_titanfuzz_total_torch)},{int(acetest_total_torch)},{int(slate_acetest_total_torch)},{int(pathfinder_total_torch)},{int(slate_pathfinder_total_torch)},{int(titanfuzz_total_tf)},{int(slate_titanfuzz_total_tf)},{int(acetest_total_tf)},{int(slate_acetest_total_tf)},{int(pathfinder_total_tf)},{int(slate_pathfinder_total_tf)}\n"
    valid_row = f"Valid,{int(titanfuzz_valid_torch)},{int(slate_titanfuzz_valid_torch)},{int(acetest_valid_torch)},{int(slate_acetest_valid_torch)},{int(pathfinder_valid_torch)},{int(slate_pathfinder_valid_torch)},{int(titanfuzz_valid_tf)},{int(slate_titanfuzz_valid_tf)},{int(acetest_valid_tf)},{int(slate_acetest_valid_tf)},{int(pathfinder_valid_tf)},{int(slate_pathfinder_valid_tf)}\n"
    validity_ratio_row = f"Ratio (%),{titanfuzz_validity_ratio_torch:.2f},{slate_titanfuzz_validity_ratio_torch:.2f},{acetest_validity_ratio_torch:.2f},{slate_acetest_validity_ratio_torch:.2f},{pathfinder_validity_ratio_torch:.2f},{slate_pathfinder_validity_ratio_torch:.2f},{titanfuzz_validity_ratio_tf:.2f},{slate_titanfuzz_validity_ratio_tf:.2f},{acetest_validity_ratio_tf:.2f},{slate_acetest_validity_ratio_tf:.2f},{pathfinder_validity_ratio_tf:.2f},{slate_pathfinder_validity_ratio_tf:.2f}\n"
    coverage_row = f"Avg. Cov.,{titanfuzz_coverage_torch:.2f},{slate_titanfuzz_coverage_torch:.2f},{acetest_coverage_torch:.2f},{slate_acetest_coverage_torch:.2f},{pathfinder_coverage_torch:.2f},{slate_pathfinder_coverage_torch:.2f},{titanfuzz_coverage_tf:.2f},{slate_titanfuzz_coverage_tf:.2f},{acetest_coverage_tf:.2f},{slate_acetest_coverage_tf:.2f},{pathfinder_coverage_tf:.2f},{slate_pathfinder_coverage_tf:.2f}\n"

    with open('data/validity_ratios.csv', 'w') as f:
        f.write(header_row)
        f.write(total_row)
        f.write(valid_row)
        f.write(validity_ratio_row)
        f.write(coverage_row)

if __name__ == "__main__":
    main()