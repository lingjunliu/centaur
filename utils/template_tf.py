import os, sys, pickle, json
import numpy as np
import tensorflow as tf

def to_tf(x, device="cpu"):
    device = "/cpu:0" if device == "cpu" else "/gpu:0"
    with tf.device(device):
        # tensor
        if isinstance(x, np.ndarray):
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

def get_signature(api, signatures, suffix=0):
    if suffix > 0:
        api = f"{api}_{suffix}"
    
    if api not in signatures:
        raise Exception(f"No signature found for {api}")
    
    api_sig = signatures[api]    

    return api_sig


def gen_concrete_input(domain, ll, arg="", rng=np.random.default_rng(42)):
    list_of_available_dtypes = [bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.float16, np.float32, np.float64, np.complex64, np.complex128, str, np.dtype]   
    if domain in ["integer", "float", "string", "boolean", "dtype"]: # primitives and dtype
        return list_of_available_dtypes[ll[1][0]](ll[0][0]) if ll[0][0] is not None else None
    elif domain == "tensor" or domain == "tensor_list": # tensors, uses the rng passed to the function        
        # Check high > low
        if ll[2][0] > ll[2][1]: # swap them
            ll[2] = [ll[2][1], ll[2][0]]
        
        # Check if range is finite and valid
        highest_limit = np.finfo(np.float64).max
        if list_of_available_dtypes[ll[1][0]] != bool and not np.isfinite(ll[2][1] - ll[2][0]):
            # clip extremes
            if ll[2][0] < -highest_limit:
                ll[2][0] = -highest_limit
            if ll[2][1] > highest_limit:
                ll[2][1] = highest_limit
            if not np.isfinite(ll[2][1] - ll[2][0]):
                # Still not finite, so we need to adjust the range
                if -1*ll[2][0] > ll[2][1]:  # low is extreme, preserve that
                    ll[2][1] = ll[2][0] + highest_limit
                else:   # high is extreme, preserve that
                    ll[2][0] = ll[2][1] - highest_limit
        elif list_of_available_dtypes[ll[1][0]] == bool:
            if not np.isfinite(ll[2][0]):
                ll[2][0] = 1
            if not np.isfinite(ll[2][1]):
                ll[2][1] = 1
            # adjust ranges with modulo 2 for bools
            ll[2] = [ll[2][0]%2, ll[2][1]%2]
            if ll[2][0] > ll[2][1]: # swap them
                ll[2] = [ll[2][1], ll[2][0]]

        val = rng.uniform(low=ll[2][0], high=ll[2][1], size=ll[0]) if None not in ll[0] else None
        return val.astype(list_of_available_dtypes[ll[1][0]]) if val is not None else None
    elif domain == "tuple":
        # CORNER CASE: If the arg is out, the tuple is a tuple of tensors
        if arg == "out":
            return tuple([np.array([]) for x in ll[0]])
        
        return tuple([list_of_available_dtypes[ll[1][0]](x) if x is not None else None for x in ll[0]])
    elif domain == "list":
        return [list_of_available_dtypes[ll[1][0]](x) if x is not None else None for x in ll[0]]
    else:
        raise NotImplementedError(f"Not implemented for {domain} yet")

def concretize_input(abstract, signature, rng, device="cpu"):
    concrete = {
        "args": [],
        "kwargs": {},
        "inner": {}
    }
    # args
    for arg, domain in signature["args"].items():
        concrete["args"].append(to_tf(gen_concrete_input(domain, abstract[arg], arg=arg, rng=rng), device=device))
    
    # kwargs
    for arg, domain in signature["kwargs"].items():
        concrete["kwargs"][arg] = to_tf(gen_concrete_input(domain, abstract[arg], arg=arg, rng=rng), device=device)

    # inner if available
    if len(signature["inner"].keys()) > 0:
        concrete["inner"] = {
            "args": [],
            "kwargs": {}
        }
        # args
        for arg, domain in signature["inner"]["args"].items():
            concrete["inner"]["args"].append(to_tf(gen_concrete_input(domain, abstract[arg], arg=arg, rng=rng), device=device))

        # kwargs
        for arg, domain in signature["inner"]["kwargs"].items():
            concrete["inner"]["kwargs"][arg] = to_tf(gen_concrete_input(domain, abstract[arg], arg=arg, rng=rng), device=device)
        
    return concrete

def main():
    api = "<api>"

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    lib = "tf"

    print(f"Using library: {lib}")
    inputs_file = os.path.join(cur_dir, f"{api}_{lib}_inputs.pkl")
    signatures_file = os.path.join(cur_dir, "signatures.json")

    with open(signatures_file, "r") as f:
        original_signatures = json.load(f)

    with open(inputs_file, "rb") as f:
        inputs = pickle.load(f)
    
    print(f"Loaded {len(inputs)} inputs from {inputs_file}")

    total = len(inputs)
    count = 0
    for cur_input in inputs:
        abs_inp = cur_input[1]
        seed = int(cur_input[2])
        suffix = int(cur_input[3])

        sig = get_signature(api, original_signatures, suffix=suffix)
        rng = np.random.default_rng(seed)
        inp = concretize_input(abs_inp, sig, rng, device="cpu")
        
        tf.config.experimental.enable_op_determinism()
        tf.random.set_seed(42)

        # API Call
        try:
            result = <api>(*inp["args"], **inp["kwargs"])

            if callable(result):
                result = result(*inp["inner"]["args"], **inp["inner"]["kwargs"])
        except Exception as e:
            pass
    
        count += 1
        print(f"Done {count}/{total} inputs")

if __name__ == "__main__":
    main()