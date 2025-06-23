import sys, os, subprocess
from utils.new_api_utils import get_lib_version
from utils.misc import get_tmp_dir
from eval.titanfuzz.monkey_patching import replace_function_invocation

def patch_code(code, api, output_dir):
    prefix = f"""
    import os, pickle, torch

    def monkey(self, func, func_str, *args, **kwargs):
        input_dict = {
            'args': args,
            'kwargs': kwargs
        }
        pkl_file = os.path.join('{output_dir}', os.path.basename(__file__)[:-2] + 'pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(input_dict, f)

        return func(*args, **kwargs)

    """
    code = replace_function_invocation(code, api, 'monkey')
    return prefix + code

def driver(api, output_dir):
    driver_code = f"""
    import os, pickle, torch
    
    dir = '{output_dir}'
    total = 0
    valid = 0
    invalid = 0
    
    for file in os.listdir(dir):
        if not file.endswith('.pkl'):
            continue
            
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

    lib = "torch"
    categories = ['non_crash'] # Add more categories as needed: crash, invalid, samples, timeout, non_crash
    
    api = get_lib_version(api, lib=lib)
    
    output_dir = os.path.join(get_tmp_dir(), "acetest_patched", api)
    os.makedirs(output_dir)
    
    for category in categories:
        result_dir = os.path.join(dir, api, category)
        if not os.path.exists(result_dir):
            continue
        
        for file in os.listdir(result_dir):
            if not file.endswith('.py'):
                continue
            
            file_path = os.path.join(result_dir, file)
            with open(file_path, 'r') as f:
                code = f.read()
                patched_code = patch_code(code, api, output_dir)
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
                    
        driver_file = os.path.join(output_dir, "driver.py")
        driver_code = driver(api, output_dir)
        with open(driver_file, 'w') as f:
            f.write(driver_code)

if __name__ == "__main__":
    main()