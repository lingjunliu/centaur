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

def get_all_sigs(torch_api):
    import torch
    all_sigs = []
    selected_sigs = []
    try:
        func = f"{torch_api}.__code__.co_varnames"
        argline = eval(func)
        selected_sigs = [list(argline)]
    except:
        func = f"{torch_api}.__doc__"
        doc = eval(func)
        for line in doc.splitlines():
            if "(" in line:
                pattern = re.search(r'\(([^)]*)\)', line)
                if pattern is not None:
                    argline = pattern.group(1)
                    cur_sig = [x.strip().split('=')[0] for x in argline.split(',')]
                    if '*' in cur_sig:
                        cur_sig.remove('*')
                    all_sigs.append(cur_sig)
                    if "->" in line:
                        selected_sigs.append(cur_sig)
        
        if len(selected_sigs) == 0:
            selected_sigs = all_sigs

        selected_sigs = sorted(selected_sigs, key=len)

    return selected_sigs


def get_arglist(torch_api, n_args=0):
    selected_sigs = get_all_sigs(torch_api)

    if len(selected_sigs) == 1:
        return selected_sigs[0]

    max_sig = []
    for sig in selected_sigs:
        if len(sig) >= n_args:
            return sig
        elif len(sig) > len(max_sig):
            max_sig = sig
    
    return max_sig