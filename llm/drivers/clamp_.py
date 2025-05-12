import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    min_val = input_dict.get("min", None)
    max_val = input_dict.get("max", None)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if min_val is not None:
            min_val = torch.tensor(min_val).cuda()
        if max_val is not None:
            max_val = torch.tensor(max_val).cuda()

    result = input_tensor.clamp_(min=min_val, max=max_val)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        min_val = input_dict.get("min", None)
        max_val = input_dict.get("max", None)

        if min_val is not None:
            min_val = tf.constant(min_val, dtype=input_tensor.dtype)
            input_tensor = tf.maximum(input_tensor, min_val)
        if max_val is not None:
            max_val = tf.constant(max_val, dtype=input_tensor.dtype)
            input_tensor = tf.minimum(input_tensor, max_val)
            
        result = input_tensor.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-0.2, 0.3, -0.8, 1.2, 0.5], dtype=np.float32),
        "min": 0.0,
        "max": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-0.2, 0.3, -0.8, 1.2, 0.5], dtype=np.float32),
        "min": 0.0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([-0.2, 0.3, -0.8, 1.2, 0.5], dtype=np.float32),
        "max": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()