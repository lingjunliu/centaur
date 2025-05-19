import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    dtype = input_dict["dtype"]

    if not cpu:
        torch.cuda.set_device(0)
    
    torch.set_autocast_cpu_dtype(dtype)
    
    result = torch.get_autocast_cpu_dtype()
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    dtype = input_dict["dtype"]
    
    return {"result": dtype}

def main():
    A_TOL = 0.01

    input_data = {
        "dtype": torch.float16
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    input_data = {
        "dtype": torch.bfloat16
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()