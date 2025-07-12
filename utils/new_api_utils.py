import torch, importlib, os, json
import numpy as np
import inspect
import tensorflow as tf

from utils.misc import map_torch_to_driver, read_file_in_root, save_file_in_root

def get_original_signatures():
    """
        Returns a dictionary containing signatures for each supported API.
        The "inner" key is only present for apis that return callable functions.
        Format: { "API": { 
                            "args": {
                                "arg": "domain",
                                ...
                            },
                            "kwargs": {...},
                            "inner": {
                                "args" : {...},
                                "kwargs": {...}
                            }
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

def get_lib_version(api, lib="torch"):
    if lib == "torch" and not api.startswith("torch"):
        _, driver_to_torch = map_torch_to_driver()
        return driver_to_torch[api] if api in driver_to_torch else api
    return api

def get_n_variations(api, lib="torch"):
    """
    Get number of variations of signatures for a specific api.
    """
    api = get_lib_version(api, lib=lib)
    signatures = get_original_signatures()
    count = 0
    if api in signatures.keys():
        return count+1
    
    while f"{api}_{count+1}" in signatures.keys():
        count += 1

    return count

def get_api_suffix(variant):
    tokens = variant.rsplit('_', 1)
    if len(tokens) == 2 and tokens[1].isdigit():
        return tokens[0], int(tokens[1])
    else:
        return variant, 0

def get_signature(api, lib="torch", suffix=0):
    """
        Returns a simplified dictionary containing all params for a supported API.
        For apis with multiple signature, suffix needs to be passed to indicate which
        signature to use.
        Format: { 
                    "arg": "domain",
                    ...
                }
    """
    api = get_lib_version(api, lib=lib)

    if suffix > 0:
        api = f"{api}_{suffix}"
    
    signatures = get_original_signatures()
    if api not in signatures:
        raise Exception(f"No signature found for {api}")
    
    api_sig = signatures[api]
    simple_sig = {}
    
    # args
    for arg, domain in api_sig["args"].items():
        simple_sig[arg] = domain

    # kwargs
    for arg, domain in api_sig["kwargs"].items():
        simple_sig[arg] = domain

    # inner if available
    if len(api_sig["inner"].keys()) > 0:
        # args
        for arg, domain in api_sig["inner"]["args"].items():
            simple_sig[arg] = domain

        # kwargs
        for arg, domain in api_sig["inner"]["kwargs"].items():
            simple_sig[arg] = domain

    return simple_sig

def match_signature(api, signature, lib="torch"):
    """
    Given a simplified signature, match the exact signature variation that was
    used to generate this.
    """
    api = get_lib_version(api, lib=lib)
    signatures = get_original_signatures()
    
    if api in signatures:
        return signatures[api]
    
    n_sigs = get_n_variations(api, lib=lib)
    for i in range(1, n_sigs+1):
        candidate_simple_sig = get_signature(api, lib=lib, suffix=i)
        if candidate_simple_sig == signature:
            k = f"{api}_{i}"
            if k in signatures:
                return signatures[k]
        
    # No signatures matched
    raise Exception(f"No matching signatures found for {api} with the simple signature {signature}")

def match_signature_to_input(input_dict, signature, match_type=False):
    for arg, value in input_dict.items():
        if arg not in signature:
            return False
        if match_type:
            if arg == "out" and value is None:
                continue  # out can be None, so we skip it
            if signature[arg] in ["tensor", "tensor_list"]:
                if not isinstance(value, (torch.Tensor, np.ndarray, tf.constant)):
                    return False
            elif signature[arg] == "dtype":
                if not isinstance(value, (np.dtype, torch.dtype, tf.dtypes.DType, type)):
                    return False
            elif signature[arg] == "string":
                if not isinstance(value, str):
                    return False
            elif signature[arg] == "integer":
                if not isinstance(value, (int, np.integer)):
                    return False
            elif signature[arg] == "float":
                if not isinstance(value, (int, np.integer, float, np.floating)):
                    return False
            elif signature[arg] == "boolean":
                if not isinstance(value, (bool, np.bool_)):
                    return False
            elif signature[arg] == "tuple":
                if not isinstance(value, (tuple, list)):
                    return False
            elif signature[arg] == "list":
                if not isinstance(value, (list, np.ndarray)):
                    return False
    
    return True

def get_signature_of_input(api, input_dict, lib="torch"):
    """
    Given an input dict, match the exact signature variation that was
    used to generate this.
    """
    api = get_lib_version(api, lib=lib)
    signatures = get_original_signatures()
    
    if api in signatures:
        return signatures[api]
    args = input_dict.keys()
    n_sigs = get_n_variations(api, lib=lib)
    for i in range(1, n_sigs+1):
        candidate_simple_sig = get_signature(api, lib=lib, suffix=i)
        match = True
        for arg in args:
            if arg not in candidate_simple_sig.keys():
                match = False
                break
        
        if match:
            return signatures[f"{api}_{i}"]
        
    # No signatures matched
    raise Exception(f"No matching signatures found for {api} with input {input_dict}")

def get_func(api, lib="torch"):
    api = get_lib_version(api, lib=lib)

    if lib == "tf" and api.startswith("tf."):
        api = api.replace("tf.", "tensorflow.")

    # Split module path and function name
    module_path, func_name = api.rsplit('.', 1)

    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise ValueError(f"Invalid module '{module_path}' in API name.")

    # Get function from the module
    func = getattr(module, func_name, None)
    if func is None:
        raise ValueError(f"Invalid function '{func_name}' in module '{module_path}'.")
    
    return func

def to_torch(x, device="cpu"):
    # tensor
    if isinstance(x, np.ndarray):
        return torch.tensor(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, tf.Tensor):
        # Convert TensorFlow tensor to PyTorch tensor
        return torch.from_numpy(x.numpy()).to(device)
    # dtype
    elif isinstance(x, np.dtype):
        return torch.tensor(np.array([], dtype=x)).dtype
    # tensor_list
    elif isinstance(x, list):
        ret_x = []
        for elem in x:
            ret_x.append(to_torch(elem, device=device))
        return ret_x
    elif isinstance(x, tuple):
        return tuple(to_torch(list(x), device=device))
    
    return x

def to_tf(x, device="cpu"):
    device = "/cpu:0" if device == "cpu" else "/gpu:0"
    with tf.device(device):
        # tensor
        if isinstance(x, torch.Tensor):
            return tf.constant(x.cpu().numpy())
        elif isinstance(x, np.ndarray):
            return tf.constant(x)
        # dtype
        elif isinstance(x, np.dtype):
            # Convert numpy dtype to tensorflow dtype using TensorFlow's built-in conversion
            return tf.dtypes.as_dtype(x)
        # tensor_list
        elif isinstance(x, list):
            ret_x = []
            for elem in x:
                ret_x.append(to_tf(elem))
            return ret_x
        elif isinstance(x, tuple):
            return tuple(to_tf(list(x)))
        
        return x

def to_numpy(x, device="cpu"):
    # tensor
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy(force=True)
    elif isinstance(x, tf.Tensor):
        # Convert TensorFlow tensor to numpy array
        return x.numpy()
    # dtype
    elif isinstance(x, torch.dtype):
        return torch.tensor([], dtype=x).numpy(force=True).dtype
    elif isinstance(x, tf.dtypes.DType):
        return tf.constant([], dtype=x).numpy().dtype
    # tensor_list
    elif isinstance(x, list):
        ret_x = []
        for elem in x:
            ret_x.append(to_numpy(elem, device=device))
        return ret_x
    elif isinstance(x, tuple):
        return tuple(to_numpy(list(x), device=device))
    
    return x

def get_input(api, input_dict, cpu=True, lib="torch"):
    """
    Given an input dictionary, returns a dictionary with the actual values
    using data structure from the library. It will also convert the dictionary
    from the simplified format to the format with args, kwargs, and inner.
    E.g. if input_dict contains data in numpy format, it will convert it to
    torch tensors if lib is torch.
    """
    api = get_lib_version(api, lib=lib)
    to_lib = to_torch if lib == "torch" else to_tf
    device = "cpu" if cpu else "cuda"
    original_signature = get_signature_of_input(api, input_dict, lib=lib)
    true_input = {
        "args": [],
        "kwargs": {},
        "inner": {}
    }
    
    # args
    for arg in original_signature["args"].keys():
        true_input["args"].append(to_lib(input_dict[arg], device=device))

    # kwargs
    for arg in original_signature["kwargs"].keys():
        true_input["kwargs"][arg] = to_lib(input_dict[arg], device=device)

    # inner if available
    if len(original_signature["inner"].keys()) > 0:
        true_input["inner"] = {
            "args": [],
            "kwargs": {}
        }
        # args
        for arg in original_signature["inner"]["args"].keys():
            true_input["inner"]["args"].append(to_lib(input_dict[arg], device=device))

        # kwargs
        for arg in original_signature["inner"]["kwargs"].keys():
            true_input["inner"]["kwargs"][arg] = to_lib(input_dict[arg], device=device)
    
    return true_input

def run_api(api, input_dict, cpu=True, lib="torch"):
    """
    Keeps the classic format of input and output intact.
    api: Takes in the name of an api (library version or base name).
    signature: In classic simplified format
    input_dict: In classic format, following the signature passed here
    cpu: True means device=cpu, False means device=cuda
    lib: torch or tf
    """
    api = get_lib_version(api, lib=lib)
    func = get_func(api, lib=lib)
    inp = get_input(api, input_dict, cpu=cpu, lib=lib)
    tf_device = "/cpu:0" if cpu else "/gpu:0"
    
    if lib == "torch":            
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
    elif lib == "tf":
        tf.config.experimental.enable_op_determinism()

    if lib == "torch":
        result = func(*inp["args"], **inp["kwargs"])
    elif lib == "tf":
        with tf.device(tf_device):
            result = func(*inp["args"], **inp["kwargs"])

    if callable(result):
        if len(inp["inner"]) == 0:
            raise Exception(f"{api} returns a function, but the input does not have inner values")
        
        if lib == "torch" and not cpu:
            result = result.cuda()
        
        if lib == "torch":
            result = result(*inp["inner"]["args"], **inp["inner"]["kwargs"])
        elif lib == "tf":
            with tf.device(tf_device):
                result = result(*inp["inner"]["args"], **inp["inner"]["kwargs"])

    result_dict = {}
    if isinstance(result, tuple) or isinstance(result, list):
        if isinstance(result, tuple):
            result = list(result)
        suffix = 0
        for elem in result:
            suffix += 1
            result_dict[f"result_{suffix}"] = to_numpy(elem)
    else:
        result_dict["result"] = to_numpy(result)

    return result_dict

def get_doc_tf(api):
    func = get_func(api, lib="tf")
    signature = f"{api}{str(inspect.signature(func))}"
    return signature + '\n' + func.__doc__ if func else None