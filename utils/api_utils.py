import importlib, os, json, re

# get the driver code for corresponding api using the name of the api
def get_driver(api, lib="torch", module="drivers"):
    # driver
    api = importlib.import_module(f"{module}.{api}")
    functions = dir(api)
    for function_name in functions:
        if lib.lower() in ["torch", "pytorch"]:
            if function_name.startswith("torch_") or function_name.startswith("pytorch_") or function_name.startswith("pt_"):
                lib_version = getattr(api, function_name)
                break
        elif lib.lower() in ["tf", "tensorflow"]:
            if function_name.startswith("tensorflow_") or function_name.startswith("tf_"):
                lib_version = getattr(api, function_name)
                break
            
    if lib_version is None:
        raise NotImplementedError(f"{api} not supported yet for {lib}")
    
    return lib_version

def get_signatures():
    """
        Returns a dictionary containing signatures for each supported API.
        Format: { "API": { 
                            "arg": "domain",
                            ...
                         } 
                  ...
                }
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    signature_file = os.path.join(cur_dir, "../signatures.json")
    signatures = {}
    with open(signature_file, "r") as f:
        signatures = json.load(f)
    
    return signatures

def get_arglist(torch_api, n_args=0):
    import torch
    try:
        func = f"{torch_api}.__code__.co_varnames"
        argline = eval(func)
        return list(argline)
    except:
        all_sigs = []
        selected_sigs = []
        func = f"{torch_api}.__doc__"
        doc = eval(func)
        for line in doc.splitlines():
            if "(" in line:
                pattern = re.search(r'\(([^)]*)\)', line)
                if pattern is not None:
                    argline = pattern.group(1)
                    all_sigs.append([x.strip().split('=')[0] for x in argline.split(',')])
                    if "->" in line:
                        selected_sigs.append([x.strip().split('=')[0] for x in argline.split(',')])
        
        if len(selected_sigs) == 0:
            selected_sigs = all_sigs
        
        max_sig = []
        for sig in selected_sigs:
            for i, arg in enumerate(sig):
                if arg.strip() == '*':
                    sig = sig[:i] + sig[i+1:]
                    break
            
            if len(sig) > n_args:
                return sig
            elif len(sig) > len(max_sig):
                max_sig = sig
        
        return max_sig