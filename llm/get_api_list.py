import torch
import inspect
import pkgutil
import importlib
import re, os
from utils.misc import api_in_file

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
        if api.endswith("torch.tensor") or api.endswith("torch.read_vitals"):
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

    driver_to_api_map = {}
    api_to_driver_map = {}
    with open(f"{CUR_DIR}/drivers_to_api.csv", "w") as f:
        f.write("Driver,API\n")
        for file in os.listdir(f"{CUR_DIR}/drivers"):
            if not file.endswith(".py"):
                continue
            driver_file = os.path.join(f"{CUR_DIR}/drivers", file)
            driver_name = file.split(".")[0]
            # check if the targeted driver is in the file
            found = False
            for api in backend_apis:
                basename = api.split('.')[-1]
                if basename.lower() == driver_name.lower() and api_in_file(api, driver_file):
                    f.write(f"{driver_name},{api}\n")
                    found = True
                    driver_to_api_map[driver_name] = api
                    api_to_driver_map[api] = driver_name
                    break
            # if not found, try all apis
            if not found:
                for api in backend_apis:
                    if api == "torch.use_deterministic_algorithms":
                        continue
                    if api_in_file(api, driver_file):
                        f.write(f"{driver_name},{api}\n")
                        driver_to_api_map[driver_name] = api
                        api_to_driver_map[api] = driver_name
                        break
    
    attempted = set()
    succeeded = set()
    with open(f"{CUR_DIR}/drivers.csv", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[0] == "driver":
                continue
            attempted.add(tokens[1])
            if tokens[-1] == "0":
                # API generation succeded
                succeeded.add(tokens[1])
            else:
                succeeded = succeeded - set(tokens[1])
    
    new_drivers = ""
    status = {}
    ex = 1
    for driver, api in driver_to_api_map.items():
        if api in succeeded:
            basename = api.split(".")[-1]
            if basename == driver:
                status[api] = "Success"
                new_drivers += f"{driver}\n"
            else:
                status[api] = "Inconsistent"
        elif api in attempted:
            status[api] = "Failed"
        else:
            status[api] = "Not Attempted"
    
    for api in supported_torch_apis:
        status[api] = "Success"
        ex += 1
 
    output_file = f'{CUR_DIR}/needs_driver.txt'
    needs = 0
    with open(output_file, 'w') as f:
        for api in sorted(backend_apis):
            if api not in status:
                status[api] = "Not Attempted"
            if status[api] == "Success":
                continue
            f.write(f"{api}\n")
            needs += 1    
    
    to_write = "API,Status\n"
    stats = {}
    for api, api_status in status.items():
        to_write += f"{api},{api_status}\n"
        if api_status not in stats:
            stats[api_status] = 0
        stats[api_status] += 1
    
    with open(f"{CUR_DIR}/driver_status.csv", "w") as f:
        f.write(to_write)

    print(f"\nStats:")
    for k, v in stats.items():
        print(f"Driver Generation: {k},{v}")

    print(f"\nSaved {needs} PyTorch APIs for which we need to create drivers to {output_file}\nExisting: {ex}")
    
    with open(f"{CUR_DIR}/new_drivers.txt", "w") as f:
        f.write(new_drivers)


if __name__ == "__main__":
    update_apis()