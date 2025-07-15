import subprocess
import sys
import os
import psutil
import time
import re    
import inspect
from bs4 import BeautifulSoup
from .misc import get_tmp_dir, create_subdir, get_dir_in_root
from .process_lcov import analyze_lcov

def monitor_memory(proc, limit=16000):
    """
    For using with subprocess.Popen, this function monitors the memory usage of the process.
    """
    memory_error = False
    
    if "MEMORY_LIMIT_COV" in os.environ.keys():
        try:
            limit = int(os.environ["MEMORY_LIMIT_COV"])
        except:
            pass

    while proc.poll() is None:
        mem_usage = psutil.Process(proc.pid).memory_info().rss / (1024 * 1024)  # Memory usage in MB

        if mem_usage > limit:
            print(f"Memory usage {mem_usage} MB exceeded limit {limit} MB. Terminating process.")
            proc.terminate()
            memory_error = True
            break
        time.sleep(0.01)  # Check every 0.01 second
    
    return memory_error

def extract_coverage_data(html_content):
    """
    Extract coverage data from HTML content.
    Returns list of tuples (path, coverage_number) and total sum.
    """
    filters = [
        # Filters are not used for now, but can be uncommented if needed
        # Tensor/memory management
        # "Copy", "Factory", "TensorShape", "TensorFactories", "TensorOperators",
        # "TensorTransform", "TensorAdvanced", "TensorCompare", "TensorProperties",
        # "Index", "Cat", "Stack", "Unfold", "Resize", "Fill", "utils", "Utils",
        # "ParamUtils", "Param", "Dispatch", "Iterator", "Stride", "Contiguous", "Type", "Tensor", "Loops",

        # # Quantization
        # "quantized", "Quant", "qconv", "qlinear", "qmatmul", "qelu", "qrelu",
        # "qsigmoid", "qtanh", "qclamp", "qthreshold", "qhardsigmoid", "qgelu",
        # "qsoftmax", "qmul", "qhardswish", "qdropout", "qnormalization", "fbgemm",
        # "qnnpack", "AffineQuantizer", "FakeQuant", "IntRepr", "MakePerTensor",

        # # Random/distribution
        # "Distribution", "Random", "Multinomial", "SobolEngine",

        # # Sampling/upsampling (typically not core compute)
        # "UpSample", "GridSampl", "FractionalMaxPool", "PixelShuffle",
        # "ChannelShuffle", "Sorting", "Histogram", "Bucketization",

        # # Sparse operations (specialized, not core dense compute)
        # "Sparse", "sparse",

        # # Infrastructure
        # "Shim", "Fallback", "Legacy", "Verbose", "Test", "Debug"
    ]
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all rows with class 'light-row'
    rows = soup.find_all('tr', class_='light-row')

    coverage_data = []
    total_coverage = 0

    for row in rows:
        # Find all td elements in the row
        tds = row.find_all('td')

        if len(tds) >= 4:  # Make sure we have at least 4 columns
            # First column: extract the path from the <a> tag
            first_td = tds[0]
            link = first_td.find('a')
            if link:
                # Get the text content of the link (the path)
                path = link.get_text().strip()

                # if path in filter, than do not add to coverage data
                flag = 0
                for filter in filters:
                    if filter in path:
                        flag = 1
                        break
                if (flag == 1):
                    continue


                # Fourth column: extract the coverage number
                fourth_td = tds[4]
                pre_tag = fourth_td.find('pre')
                if pre_tag:
                    # Extract text like "0.00% (0/34)" and get the first number before "/"
                    coverage_text = pre_tag.get_text().strip()

                    # Use regex to find the pattern (number/number)
                    match = re.search(r'\((\d+)/\d+\)', coverage_text)
                    if match:
                        coverage_number = int(match.group(1))
                        coverage_data.append((path, coverage_number))
                        total_coverage += coverage_number

    return coverage_data, total_coverage

def gen_cov(cmd_line, lib="torch", prefix="default", capture_output=True, gen_lcov=True, gen_html=False, gen_text=False, native_only=False):
    """
    Generate coverage data after running a command. To differentiate the generated profraw and profdata files from other
    parallel executions, provide a prefix for the file names. The default is "default".
    capture_output=True will print the output (default behavior).
    
    Example: gen_cov("python -m eval.patched_drivers.GroupNorm_cov_in_loop", lib="torch", prefix="GroupNorm", capture_output=True)
    This will run "python -m eval.patched_drivers.GroupNorm_cov_in_loop" and calculate coverage. It will use "GroupNorm" as the names for the profraw and profdata files. 
    """
    if lib == "torch":
        import torch
        TORCH_BUILD_DIR = os.path.dirname(inspect.getfile(torch))
        print(f"Using torch from {TORCH_BUILD_DIR}")
        LIB1 = f"{TORCH_BUILD_DIR}/lib/libtorch_cpu.so"
        # LIB2 = f"{TORCH_BUILD_DIR}/lib/libtorch.so"
    elif lib == "tf":
        import tensorflow as tf
        TF_BUILD_DIR = os.path.dirname(inspect.getfile(tf))
        print(f"Using tensorflow from {TF_BUILD_DIR}")
        LIB1 = f"{TF_BUILD_DIR}/libtensorflow_cc.so.2"
    else:
        raise Exception(f"Unsupported library {lib}, choose torch or tf")

    
    if "LLVM_BINDIR" in os.environ:
        llvm_prefix = f'{os.environ["LLVM_BINDIR"]}/'
    else:
        llvm_prefix = ""
    
    cov_dir = create_subdir(get_tmp_dir(), "coverage_raw_files")
    profraw_file = f"{cov_dir}/{prefix}.profraw"
    profdata_file = f"{cov_dir}/{prefix}.profdata"

    # cleanup
    if os.path.isfile(profraw_file):
        os.remove(profraw_file)

    if os.path.isfile(profdata_file):
        os.remove(profdata_file)

    # call
    try:
        custom_env = os.environ.copy()
        custom_env["LLVM_PROFILE_FILE"] = profraw_file
        
        return_obj = subprocess.run(cmd_line.split(), capture_output=capture_output, env=custom_env)
        # memory_error = monitor_memory(return_obj) # Use with subprocess.Popen if you want to monitor memory usage
    except subprocess.CalledProcessError as err:
        raise Exception(f"Could not run {cmd_line}. Error Code {err.returncode}: {err}")
    except KeyboardInterrupt:
        print("Stopped...")
        raise KeyboardInterrupt    

    return_code = return_obj.returncode

    if capture_output:
        print(return_obj.stdout.decode())

    if len(return_obj.stderr.decode()) > 0:
        print(f"Error faced while running code: {return_obj.stderr.decode()}")

    # coverage
    if not os.path.isfile(profraw_file):
        raise Exception(f"{profraw_file} missing")

    try:
        return_obj = subprocess.run(
            [f"{llvm_prefix}llvm-profdata", "merge", "-sparse", profraw_file, "-o", profdata_file],
            capture_output=capture_output,
        )
    except subprocess.CalledProcessError as err:
        raise Exception(f"Could not generate {profdata_file}. Error Code {err.returncode}: {err}")
    except KeyboardInterrupt:
        print("Stopped...")
        raise KeyboardInterrupt
    
    if not os.path.isfile(profdata_file):
        raise Exception(f"{profdata_file} missing")
    
    if len(return_obj.stderr.decode()) > 0:
        print(f"Error faced while running llvm-profdata: {return_obj.stderr.decode()}")

    if gen_lcov:
        try:
            return_obj = subprocess.run(
                [
                    f"{llvm_prefix}llvm-cov",
                    "export",
                    f"-instr-profile={profdata_file}",
                    "-format=lcov",
                    "-object",
                    LIB1,
                    # LIB2,
                ],
                capture_output=True,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(f"Could not generate lcov data. Error Code {err.returncode}: {err}")
        except KeyboardInterrupt:
            print("Stopped...")
            raise KeyboardInterrupt

        lcov_data = return_obj.stdout
        lcov_data = lcov_data.decode()

        if len(return_obj.stderr.decode()) > 0:
            print(f"Error faced while running llvm-cov: {return_obj.stderr.decode()}")
    else:
        lcov_data = {}

    if gen_html:
        print("Generating HTML coverage report...")
        try:
            cmd_html =  [
                            f"{llvm_prefix}llvm-cov",
                            "show",
                            LIB1,
                            f"-instr-profile={profdata_file}",
                            "-format=html",
                            "-show-branches=count",
                            f"-output-dir={cov_dir}/{prefix}"
                        ]
            instrumentation_dir = get_dir_in_root('instrumented_pytorch')
            if native_only:                                
                filter_dir = f"{instrumentation_dir}/pytorch/aten/src/ATen/native/"                
                print(f"Filtering to only the native folder at {filter_dir}")
            else:
                filter_dir = f"{instrumentation_dir}/pytorch/"
            
            cmd_html.append(filter_dir)

            return_obj = subprocess.run(cmd_html, capture_output=True)
        except subprocess.CalledProcessError as err:
            raise Exception(f"Could not generate html data. Error Code {err.returncode}: {err}")
        except KeyboardInterrupt:
            print("Stopped...")
            raise KeyboardInterrupt
        
        if len(return_obj.stderr.decode()) > 0:
            print(f"Error faced while generating html: {return_obj.stderr.decode()}")


    if gen_text:    
        print("Generating Text coverage report...")
        try:
            return_obj = subprocess.run(
                [
                    f"{llvm_prefix}llvm-cov",
                    "show",
                    f"-instr-profile={profdata_file}",
                    "-format=text",
                    "-show-branches=count",
                    f"-output-dir={cov_dir}/{prefix}",
                    "-object",
                    LIB1,
                    # LIB2,
                ],
                capture_output=True,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(f"Could not generate text data. Error Code {err.returncode}: {err}")
        except KeyboardInterrupt:
            print("Stopped...")
            raise KeyboardInterrupt
        
        if len(return_obj.stderr.decode()) > 0:
            print(f"Error faced while generating text: {return_obj.stderr.decode()}")

    # cleanup
    if os.path.isfile(profraw_file):
        os.remove(profraw_file)

    if os.path.isfile(profdata_file):
        os.remove(profdata_file)
    
    return return_code, lcov_data

def get_coverage_numbers(cmd_line, lib="torch", prefix="default", capture_output=True, gen_lcov=True, gen_html=False, gen_text=False, save_lcov=False, native_only=False):
    """
    Generate # of branches and # of lines covered in Pytorch after running a command.
    To differentiate the generated profraw and profdata files from other
    parallel executions, provide a prefix for the file names. The default is "default".
    capture_output=True will print the output (default behavior).
    
    Example: get_coverage_numbers("python -m eval.patched_drivers.GroupNorm_cov_in_loop", lib="torch", prefix="GroupNorm", capture_output=True)
    This will run "python -m eval.patched_drivers.GroupNorm_cov_in_loop" and calculate coverage. It will use "GroupNorm" as the names for the profraw and profdata files.
    It will return the num_branches, num_lines, return_code of executing cmd_line and a dict containing detailed information.
    """
    return_code, lcov_data = gen_cov(cmd_line, lib=lib, prefix=prefix, capture_output=capture_output, gen_lcov=gen_lcov, gen_html=gen_html, gen_text=gen_text, native_only=native_only)

    cov_dir = create_subdir(get_tmp_dir(), "coverage_raw_files")
    if save_lcov:
        lcov_file = os.path.join(cov_dir, f"{prefix}.lcov")
        with open(lcov_file, "w") as f:
            f.write(lcov_data)
    
    num_branches = 0
    num_lines = 0
    coverage_dict = None
    
    if gen_html:    # Use HTML to extract branch coverage number
        html_file = f"{cov_dir}/{prefix}/index.html"
        if os.path.isfile(html_file):
            with open(html_file, "r") as f:
                html_content = f.read()
            coverage_dict, num_branches = extract_coverage_data(html_content)
    elif gen_lcov:   # Use lcov data to extract branch and line coverage numbers
        coverage_dict = analyze_lcov(lcov_data)
        for filename, coverage_info in coverage_dict.items():
            num_branches += len(coverage_info["branches"])
            num_lines += len(coverage_info["lines"])
    else:
        raise Exception("Only supports html or lcov to compute branch and line coverage")
        
    return num_branches, num_lines, return_code, coverage_dict
    

def main():
    if len(sys.argv) > 1:
        cmd_line = sys.argv[1]
        lib = sys.argv[2] if len(sys.argv) > 2 else "torch"
        gen_html = sys.argv[3].lower() == "html" if len(sys.argv) > 3 else False
        native_only = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False
        prefix = sys.argv[5] if len(sys.argv) > 5 else "default"
        num_branches, num_lines, return_code, coverage_dict = get_coverage_numbers(cmd_line, lib=lib, prefix=prefix, gen_html=gen_html, gen_lcov=not gen_html, native_only=native_only)

        if return_code != 0:
            print(f"Executing {cmd_line} failed with return code {return_code}")
        else:
            print(f"Execution successful")
        print(f"Number of branches covered: {num_branches}")
        print(f"Number of lines covered: {num_lines}")

if __name__ == "__main__":
    main()