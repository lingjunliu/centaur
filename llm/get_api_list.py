import torch
import inspect
import pkgutil
import importlib
import re, os
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

    backend_apis = []

    for api in torch_apis:
        if pattern.match(api):
            backend_apis.append(api)

    # Write to file
    output_file = 'api_full.txt'
    with open(output_file, 'w') as f:
        for api in sorted(backend_apis):
            f.write(f"{api}\n")

    print(f"Saved {len(backend_apis)} PyTorch APIs to {output_file}")

    # Write apis for which we need to create drivers
    supported_file = 'supported_apis.txt'
    with open(supported_file, 'r') as f:
        supported_apis = set(line.strip() for line in f)

    output_file = 'needs_driver.txt'
    needs = 0
    with open(output_file, 'w') as f:
        for api in sorted(backend_apis):
            basename = api.split('.')[-1]
            if basename in supported_apis:
                continue
            if os.path.exists(f"drivers/{basename}.py"):
                continue
            f.write(f"{api}\n")
            needs += 1

    print(f"Saved {needs} PyTorch APIs for which we need to create drivers to {output_file}")
    
if __name__ == "__main__":
    update_apis()