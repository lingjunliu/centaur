import subprocess
import sys
import os
import psutil
import time
from .misc import get_tmp_dir, create_subdir
from .process_lcov import analyze_lcov

def monitor_memory(proc, limit=16000):
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

def gen_cov_torch(cmd_line, prefix="default", capture_output=True, gen_html=False):
    """
    Generate coverage data after running a command. To differentiate the generated profraw and profdata files from other
    parallel executions, provide a prefix for the file names. The default is "default".
    capture_output=True will print the output (default behavior).
    
    Example: gen_cov_torch("-m eval.patched_drivers.GroupNorm_cov_in_loop", prefix="GroupNorm", capture_output=True)
    This will run "python -m eval.patched_drivers.GroupNorm_cov_in_loop" and calculate coverage. It will use "GroupNorm" as the names for the profraw and profdata files. 
    """
    if "TORCH_BUILD_DIR" in os.environ:
        TORCH_BUILD_DIR = os.environ["TORCH_BUILD_DIR"]
    else:
        print("WARNING: TORCH_BUILD_DIR environment variable was not set")
        import inspect
        import torch
        TORCH_BUILD_DIR = os.path.dirname(inspect.getfile(torch))
        print(f"Using torch from {TORCH_BUILD_DIR}")
    
    if "LLVM_BINDIR" in os.environ:
        llvm_prefix = f'{os.environ["LLVM_BINDIR"]}/'
    else:
        llvm_prefix = ""

    LIB1 = f"{TORCH_BUILD_DIR}/lib/libtorch_cpu.so"
    LIB2 = f"{TORCH_BUILD_DIR}/lib/libtorch.so"
    
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
        
        return_obj = subprocess.Popen(cmd_line.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=custom_env)
        memory_error = monitor_memory(return_obj)
    except subprocess.CalledProcessError as err:
        raise Exception(f"Could not run {cmd_line}. Error Code {err.returncode}: {err}")
    except KeyboardInterrupt:
        print("Stopped...")
        raise KeyboardInterrupt    

    return_code = return_obj.returncode

    if capture_output:
        print(return_obj.communicate()[0].decode())

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

    try:
        return_obj = subprocess.run(
            [
                f"{llvm_prefix}llvm-cov",
                "export",
                f"-instr-profile={profdata_file}",
                "-format=lcov",
                "-object",
                LIB1,
                LIB2,
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
        
    if gen_html:
        print("Generating HTML coverage report...")
        try:
            return_obj = subprocess.run(
                [
                    f"{llvm_prefix}llvm-cov",
                    "show",
                    f"-instr-profile={profdata_file}",
                    "-format=html",
                    "-show-branches=count",
                    "-coverage-watermark=2,1",
                    f"-output-dir={cov_dir}/{prefix}",
                    "-object",
                    LIB1,
                    LIB2,
                ],
                capture_output=True,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(f"Could not generate html data. Error Code {err.returncode}: {err}")
        except KeyboardInterrupt:
            print("Stopped...")
            raise KeyboardInterrupt
        
        if len(return_obj.stderr.decode()) > 0:
            print(f"Error faced while generating html: {return_obj.stderr.decode()}")
        
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
                    LIB2,
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
    
    return return_code, lcov_data, memory_error

def get_cov_torch(cmd_line, prefix="default", capture_output=True, gen_html=False, save_lcov=False):
    """
    Generate # of branches and # of lines covered in Pytorch after running a command.
    To differentiate the generated profraw and profdata files from other
    parallel executions, provide a prefix for the file names. The default is "default".
    capture_output=True will print the output (default behavior).
    
    Example: gen_cov_torch("-m eval.patched_drivers.GroupNorm_cov_in_loop", prefix="GroupNorm", capture_output=True)
    This will run "python -m eval.patched_drivers.GroupNorm_cov_in_loop" and calculate coverage. It will use "GroupNorm" as the names for the profraw and profdata files.
    It will return the num_branches, num_lines, return_code of executing cmd_line and a dict containing detailed information.
    """
    return_code, lcov_data, memory_error = gen_cov_torch(cmd_line, prefix=prefix, capture_output=capture_output, gen_html=gen_html)
        
    if memory_error and capture_output:
        print(f"WARNING: Faced memory error while executing {cmd_line}")

    if save_lcov:
        cov_dir = create_subdir(get_tmp_dir(), "coverage_raw_files")
        lcov_file = os.path.join(cov_dir, f"{prefix}.lcov")
        with open(lcov_file, "w") as f:
            f.write(lcov_data)
    
    coverage_dict = analyze_lcov(lcov_data)
    num_branches = 0
    num_lines = 0
    for filename, coverage_info in coverage_dict.items():
        num_branches += len(coverage_info["branches"])
        num_lines += len(coverage_info["lines"])
        
    return num_branches, num_lines, return_code, coverage_dict
    

def main():
    if len(sys.argv) > 1:
        cmd_line = sys.argv[1]
        gen_html = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
        prefix = sys.argv[3] if len(sys.argv) > 3 else "default"
        num_branches, num_lines, return_code, coverage_dict = get_cov_torch(cmd_line, prefix=prefix, gen_html=gen_html)

        if return_code != 0:
            print(f"Executing {cmd_line} failed with return code {return_code}")
        else:
            print(f"Execution successful")
        print(f"Number of branches covered: {num_branches}")
        print(f"Number of lines covered: {num_lines}")

if __name__ == "__main__":
    main()