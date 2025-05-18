import torch
import inspect
import pkgutil
import importlib
import re, os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_apis(module, prefix=''):
    apis = []
    
    # Get all callable attributes directly in the module
    for name, obj in inspect.getmembers(module):
        if not name.startswith('_'):  # Skip private/internal APIs
            if callable(obj) or inspect.isclass(obj) or inspect.ismodule(obj):
                apis.append(f"{prefix}{name}")
    
    # Get submodules recursively
    for _, name, is_pkg in pkgutil.iter_modules(module.__path__, module.__name__ + '.'):
        if not name.split('.')[-1].startswith('_'):  # Skip private/internal modules
            try:
                submodule = importlib.import_module(name)
                if is_pkg:
                    apis.extend(get_apis(submodule, f"{name}."))
                else:
                    # We already added the module name above, so we don't need to add it again
                    pass
            except Exception as e:
                print(f"Error importing {name}: {e}")
                pass  # Skip modules that can't be imported
    
    return apis

def api_in_file(api, filename):
    """
    Check if the given API is present in the specified file.
    """
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return False
    with open(filename, "r") as f:
        for line in f.readlines():
            if f"{api}(" in line:
                return True
    return False

def update_apis():
    # Get all PyTorch APIs
    torch_apis = get_apis(torch, 'torch.')

    # Define patterns for backend-related APIs
    backend_patterns = [
                # Core tensor operations (likely implemented in at:: namespace)
                r'torch\.Tensor\.',
                r'torch\.tensor',
                r'torch\.[a-z_]+$',  # Basic tensor operations
                
                # Distributed computing (c10d namespace)
                # r'torch\.distributed',
                
                # Neural network modules (torch::nn namespace)
                r'torch\.nn\.',
                
                # Data loading (torch::data namespace)
                # r'torch\.utils\.data',
                
                # JIT/TorchScript (C++ implementation)
                r'torch\.jit',
                
                # FFT operations (likely C++ backend)
                r'torch\.fft',
                
                # Linear algebra operations (likely C++ backend)
                r'torch\.linalg',
                
                # Special functions (likely C++ backend)
                r'torch\.special',
                
                # Storage classes (C++ implementation)
                r'torch\.[A-Z][a-zA-Z]*Storage'
    ]
    combined_pattern = '|'.join(backend_patterns)
    pattern = re.compile(combined_pattern)

    backend_apis = set()

    for api in torch_apis:
        if api.endswith("torch.tensor"):
            continue
        if pattern.match(api):
            backend_apis.add(api)

    supported_file = f'{CUR_DIR}/supported_apis.txt'
    with open(supported_file, 'r') as f:
        supported_apis = set(line.strip() for line in f)
    
    supported_torch_apis = set()    
    with open(f"{CUR_DIR}/supported.csv", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[0] == "API":
                continue
            if tokens[0] in supported_apis:
                supported_torch_apis.add(tokens[1])
                
    backend_apis = backend_apis.union(supported_torch_apis)
    
    # Write to file
    output_file = f'{CUR_DIR}/api_full.txt'
    with open(output_file, 'w') as f:
        for api in sorted(backend_apis):
            f.write(f"{api}\n")

    print(f"Saved {len(backend_apis)} PyTorch APIs to {output_file}")


    attempted = set()
    overridden = set()
    inconsistent = set()
    succeeded = set()
    with open(f"{CUR_DIR}/drivers.csv", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[0] == "driver":
                continue
            if api_in_file(tokens[1], os.path.join(f"{CUR_DIR}/drivers", f"{tokens[0]}.py")):
                attempted.add(tokens[1])
                if tokens[-1] == "0":
                    # API generation succeded
                    succeeded.add(tokens[1])
            elif api_in_file(tokens[0], os.path.join(f"{CUR_DIR}/drivers", f"{tokens[0]}.py")):
                overridden.add(tokens[1])
            else:
                inconsistent.add(tokens[1])
    
    
    failed_generation = attempted - succeeded
    not_attempted = backend_apis - attempted    
    not_attempted = not_attempted - overridden    
    not_attempted = not_attempted - inconsistent
    extra = supported_torch_apis - not_attempted
    not_attempted = not_attempted - supported_torch_apis
 
    output_file = f'{CUR_DIR}/needs_driver.txt'
    needs = 0
    with open(output_file, 'w') as f:
        for api in sorted(backend_apis):
            if api in succeeded or api in supported_torch_apis:
                continue
            f.write(f"{api}\n")
            needs += 1    
    
    to_write = "API,Status\n"
    for api in sorted(backend_apis):
        if api in supported_torch_apis:
            to_write += f"{api},Driver generation successful\n"            
        elif api in failed_generation:
            to_write += f"{api},Driver generation tried but failed\n"            
        elif api in succeeded:
            to_write += f"{api},Driver generation successful\n"
        elif api in overridden:
            to_write += f"{api},Driver generation overridden\n"
        elif api in inconsistent:
            to_write += f"{api},Driver generation inconsistent\n"
        elif api in not_attempted:
            to_write += f"{api},Driver generation not attempted\n"            
        else:
            to_write += f"{api},Unknown\n"
    
    with open(f"{CUR_DIR}/driver_status.csv", "w") as f:
        f.write(to_write)            

    print(f"\nStats:\nDriver generation attempted,{len(attempted)}\nSucceeded,{len(succeeded)}\nFailed,{len(failed_generation)}\nOverridden,{len(overridden)}\nInconsistent,{len(inconsistent)}\nPreviously existed,{len(supported_torch_apis)}\nNot attempted,{len(not_attempted)}\nRetried generation despite existing,{len(extra)}\nTotal,{len(backend_apis)}")

    print(f"\nSaved {needs} PyTorch APIs for which we need to create drivers to {output_file}\n")
    
    with open(f"{CUR_DIR}/other_bugs.txt", "r") as f:
        for line in f.readlines():
            api = line.strip()
            if api in succeeded:
                print(f"{api},Succeded")
            elif api in failed_generation:
                print(f"{api},Failed")
            elif api in overridden:
                print(f"{api},Overridden")
            elif api in inconsistent:
                print(f"{api},Inconsistent")
            elif api in supported_torch_apis:
                print(f"{api},Previously Supported")
            else:
                print(f"{api},Unsupported")
    
    with open(f"{CUR_DIR}/drivers_to_api.csv", "w") as f:
        f.write("Driver,API\n")
        for file in os.listdir(f"{CUR_DIR}/drivers"):
            if not file.endswith(".py"):
                continue
            driver_file = os.path.join(f"{CUR_DIR}/drivers", file)
            driver_name = file.split(".")[0]
            for api in backend_apis:
                if api_in_file(api, driver_file):
                    f.write(f"{driver_name},{api}\n")
                    break


if __name__ == "__main__":
    update_apis()