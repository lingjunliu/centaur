import sys, os, subprocess, re
from utils.new_api_utils import get_lib_version
from utils.misc import get_tmp_dir

def replace_function_invocation(script, old_function_name, new_function_name):
    # The 're.DOTALL' flag allows '.' to match newlines.
    # We use a non-greedy quantifier (.*?) to match as little as possible.
    pattern = re.compile(rf"{old_function_name}\((.*?)\)", re.DOTALL)

    def replacer(match):
        old_args = match.group(1)
        # Ensure that old_function_name is correctly quoted if it's a string literal
        # in the new invocation, as per your example.
        return f"{new_function_name}({old_function_name}, {old_args})"

    return pattern.sub(replacer, script)

def patch_code(code, api, output_dir, lib="torch"):
    if lib == "torch":
        lib_import = "import torch"
    elif lib == "tf":
        lib_import = "import tensorflow as tf"
    else:
        raise Exception(f"Unsupported library {lib}")

    prefix = f"""
import os, pickle
{lib_import}

def monkey(func, *args, **kwargs):
    input_dict = {{
        'args': args,
        'kwargs': kwargs
    }}
    pkl_file = os.path.join('{output_dir}', os.path.basename(__file__)[:-2] + 'pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(input_dict, f)

    return func(*args, **kwargs)
"""
    code = replace_function_invocation(code, api, 'monkey')
    return prefix + code

def driver(api, output_dir, lib="torch"):
    if lib == "torch":
        lib_import = "import torch"
    elif lib == "tf":
        lib_import = "import tensorflow as tf"
    else:
        raise Exception(f"Unsupported library {lib}")

    driver_code = f"""
import os, pickle
{lib_import}

dir = '{output_dir}'
total = 0
valid = 0
invalid = 0

for file in os.listdir(dir):
    if not file.endswith('.pkl'):
        continue
    
    total += 1
    with open(os.path.join(dir, file), 'rb') as f:
        input_dict = pickle.load(f)
        try:
            output = {api}(*input_dict['args'], **input_dict['kwargs'])
            valid += 1
        except Exception as e:
            invalid += 1
            
print(total, valid, invalid)
"""
    return driver_code


def main():
    api = sys.argv[1]
    dir = sys.argv[2]
    lib = sys.argv[3] if len(sys.argv) > 3 else "torch"

    if not os.path.exists(dir):
        print(f"{dir} does not exist")
        return

    categories = ['non_crash'] # Add more categories as needed: crash, invalid, samples, timeout, non_crash
    
    api = get_lib_version(api, lib=lib)
    
    output_dir = os.path.join(get_tmp_dir(), "acetest_patched", api)
    os.makedirs(output_dir, exist_ok=True)
    
    for category in categories:
        result_dir = os.path.join(dir, api, category)
        if not os.path.exists(result_dir):
            continue
        
        files = os.listdir(result_dir)
        for i, file in enumerate(files):
            if not file.endswith('.py'):
                continue
            
            file_path = os.path.join(result_dir, file)
            with open(file_path, 'r') as f:
                code = f.read()
                patched_code = patch_code(code, api, output_dir, lib=lib)
                patched_file = os.path.join(output_dir, file)
                with open(patched_file, "w") as f:
                    f.write(patched_code)
                    
                try:
                    return_obj = subprocess.run(["python", patched_file], capture_output=True)
                except subprocess.CalledProcessError as err:
                    raise Exception(f"Could not run patched driver. Error Code {err.returncode}: {err}")
                except KeyboardInterrupt:
                    print("Stopped...")
                    raise KeyboardInterrupt
                
                if len(return_obj.stderr.decode()) > 0:
                    print(f"Error faced while running patched code: {return_obj.stderr.decode()}")
                    
                print(f"Patched {i+1}/{len(files)} files        ", end='\r', flush=True)
                    
        driver_file = os.path.join(output_dir, "driver.py")
        driver_code = driver(api, output_dir, lib=lib)
        with open(driver_file, 'w') as f:
            f.write(driver_code)

if __name__ == "__main__":
    main()