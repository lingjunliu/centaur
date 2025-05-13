import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    dtype = input_dict["dtype"]

    if not cpu:
        torch.cuda.set_device(0)
    
    torch.set_autocast_dtype(device_type='cuda' if not cpu else 'cpu', dtype=dtype)

    result = torch.get_autocast_dtype(device_type='cuda' if not cpu else 'cpu')
    
    if not cpu:
        torch.cuda.set_device('cpu')
    
    return {"result": str(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import torch
    
    dtype = input_dict["dtype"]
    
    if dtype == "torch.float16":
        tf_dtype = tf.float16
    elif dtype == "torch.bfloat16":
        tf_dtype = tf.bfloat16
    else:
        raise ValueError("Unsupported dtype for tensorflow conversion")
    
    with tf.device('/CPU:0' if cpu else '/GPU:0'):
        tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy(str(tf_dtype.name)))
        
        policy = tf.keras.mixed_precision.global_policy()
        
        result = policy.compute_dtype
        
    return {"result": str(result.name)}

def main():
    import torch
    A_TOL = 0.01

    torch.set_autocast_dtype(device_type='cpu', dtype=torch.float32)

    input_data = {
        "dtype": torch.float16
    }
    
    torch_result = torch_version({"dtype": torch.float16}, cpu=True)
    tf_result = tensorflow_version({"dtype": "torch.float16"}, cpu=True)

    assert torch_result["result"] == "torch.float32"
    assert tf_result["result"] == "float16"

    input_data = {
        "dtype": torch.bfloat16
    }
    
    torch_result = torch_version({"dtype": torch.bfloat16}, cpu=True)
    tf_result = tensorflow_version({"dtype": "torch.bfloat16"}, cpu=True)

    assert torch_result["result"] == "torch.float32"
    assert tf_result["result"] == "bfloat16"
    print("Success")

if __name__ == "__main__":
    main()