import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    min_val = input_dict.get("min", None)
    max_val = input_dict.get("max", None)

    if min_val is not None:
        min_val = torch.tensor(min_val)
    if max_val is not None:
        max_val = torch.tensor(max_val)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if min_val is not None:
            min_val = min_val.cuda()
        if max_val is not None:
            max_val = max_val.cuda()
    
    if min_val is None and max_val is None:
        result = input_tensor
    else:
        result = torch.clip(input_tensor, min=min_val, max=max_val)
    
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

        if min_val is None:
            min_val = tf.float32.min
        if max_val is None:
            max_val = tf.float32.max
        
        result = tf.clip_by_value(input_tensor, clip_value_min=min_val, clip_value_max=max_val)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, 0.5, 2.0, -3.0, 1.5], dtype=np.float32),
        "min": -0.5,
        "max": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-1.0, 0.5, 2.0, -3.0, 1.5], dtype=np.float32),
        "min": -0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-1.0, 0.5, 2.0, -3.0, 1.5], dtype=np.float32),
        "max": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-1.0, 0.5, 2.0, -3.0, 1.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()