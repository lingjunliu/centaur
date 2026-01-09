import sys, os, subprocess

from utils.coverage_utils import extract_coverage_data
from utils.misc import get_tmp_dir

def merge_profdata(dir, out_dir):
    profdata_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.profdata')]
    if not profdata_files:
        print(f"No .profdata files found in {dir} to merge.")
        return False

    merged_profdata = os.path.join(out_dir, 'merged_all.profdata')
    merge_command = ['llvm-profdata', 'merge', '-o', merged_profdata, f"{dir}/*.profdata"]
    return_obj = subprocess.run(' '.join(merge_command), shell=True, capture_output=True)

    if return_obj.returncode != 0:
        print(f"Error merging profdata files: {return_obj.stderr.decode()}")
        return False
    
    print(f"Merged profdata files into {merged_profdata}")
    for f in profdata_files:
        os.remove(f)

    return merged_profdata

def gen_html(profdata_file, out_dir):
    print("Generating HTML coverage report...")
    html_dir = os.path.join(out_dir, "html_report")
    os.makedirs(html_dir, exist_ok=True)
    try:
        cmd_html =  [
                        f"{llvm_prefix}llvm-cov",
                        "show",
                        LIB1,
                        f"-instr-profile={profdata_file}",
                        "-format=html",
                        "-show-branches=count",
                        f"-output-dir={html_dir}"
                    ]            

        return_obj = subprocess.run(cmd_html, capture_output=True)
    except subprocess.CalledProcessError as err:
        raise Exception(f"Could not generate html data. Error Code {err.returncode}: {err}")
    except KeyboardInterrupt:
        print("Stopped...")
        raise KeyboardInterrupt
    
    if len(return_obj.stderr.decode()) > 0:
        print(f"Error faced while generating html: {return_obj.stderr.decode()}")

    if os.path.isfile(profdata_file):
        os.remove(profdata_file)

    html_file = os.path.join(html_dir, "index.html")

    if not os.path.isfile(html_file):
        raise Exception("HTML coverage report generation failed.")

    with open(html_file, "r") as f:
        html_content = f.read()
    coverage_dict, num_branches = extract_coverage_data(html_content)

    print(f"Number of branches covered: {num_branches}")

    os.removedirs(html_dir)

    return coverage_dict, num_branches

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(get_tmp_dir(), "coverage_raw_files")
    out_dir = os.path.join(get_tmp_dir(), "merged_coverage")
    os.makedirs(out_dir, exist_ok=True)

    merged_profdata = merge_profdata(dir, out_dir)
    if merged_profdata:
        _, num_branches = gen_html(merged_profdata, out_dir)

    result_file = os.path.join(out_dir, "coverage_summary.csv") if not csv_file else csv_file

    if not os.path.isfile(result_file):
        print(f"Creating new result file at {result_file}")

    with open(result_file, "a") as f:
        f.write(num_branches)