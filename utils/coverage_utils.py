import pickle
import subprocess
import sys
import os
import importlib
import numpy as np
import psutil
import time

SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = "/".join(SCRIPT_DIR.split("/")[:-1])

sys.path.append(REPO_DIR)

def get_input_stats(input_file):
    is_np = True

    shape_l = []
    dtype_l = []
    min_max_l = []

    try:
        if input_file.endswith(".py"):
            spec = importlib.util.spec_from_file_location(
                name="input",
                location=input_file,
            )
            module_inp = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module_inp)

            args = module_inp.get_args()
            is_np = False
        elif input_file.endswith(".pickle"):
            with open(input_file, "rb") as fpickle:
                args_dict = pickle.load(fpickle)
            
            args = []
            for k in args_dict.keys():
                args.append(args_dict[k])
        else:
            raise Exception(f"Input file {input_file} not supported")
    except:
        return shape_l, dtype_l, min_max_l

    for arg in args:
        if hasattr(arg, "shape"):
            try:
                shape_l.append(arg.shape)
                dtype_l.append(arg.dtype)
                if 0 in list(arg.shape):
                    min_max_l.append((0, 0))
                else:
                    if is_np:
                        min_max_l.append((np.min(arg), np.max(arg)))
                    else:
                        try:
                            min_max_l.append((np.min(arg.numpy()), np.max(arg.numpy())))
                        except:
                            min_max_l.append((0, 0))
            except:
                shape_l.append("non_tensor")
                dtype_l.append(type(arg))
                if isinstance(arg, list):
                    min_max_l.append(f"length: {len(arg)}")
                else:    
                    min_max_l.append(arg)
        else:
            try:
                shape_l.append("non_tensor")
                dtype_l.append(type(arg))
                if isinstance(arg, list):
                    min_max_l.append(f"length: {len(arg)}")
                else:    
                    min_max_l.append(arg)
            except:
                continue
                        
    return shape_l, dtype_l, min_max_l

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

def gen_cov_torch(driver, input_file, cpu=True, capture_output=True, is_snippet=False):
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
    
    profraw_file = f"{os.path.dirname(input_file)}/{driver}.profraw"
    profdata_file = f"{os.path.dirname(input_file)}/{driver}.profdata"

    # cleanup
    if os.path.isfile(profraw_file):
        os.remove(profraw_file)

    if os.path.isfile(profdata_file):
        os.remove(profdata_file)

    # call
    if is_snippet:  # passed input file is the snippet to run
        cmd = ["python"] + input_file.split()
    else:           # passed input file is the input for the driver
        if os.path.isdir(input_file):
            driver_wrapper = "run_driver_in_loop.py"
        else:
            driver_wrapper = "run_driver.py"
        cmd = ["python", driver_wrapper, driver, input_file, str(cpu)]
    
    try:
        custom_env = os.environ.copy()
        custom_env["LLVM_PROFILE_FILE"] = profraw_file
        
        return_obj = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=custom_env)
        memory_error = monitor_memory(return_obj)
    except subprocess.CalledProcessError as err:
        raise Exception(f"Could not run {driver} with input {input_file}. Error Code {err.returncode}: {err}")
    except KeyboardInterrupt:
        print("Stopped...")
        raise KeyboardInterrupt    

    return_code = return_obj.returncode

    if not capture_output:
        print("Sorry! Capturing output disabled for now...")

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

    # cleanup
    if os.path.isfile(profraw_file):
        os.remove(profraw_file)

    if os.path.isfile(profdata_file):
        os.remove(profdata_file)
    
    return return_code, lcov_data, memory_error