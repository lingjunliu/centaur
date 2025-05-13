import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):

    data = input_dict["data"]
    dtype = input_dict.get("dtype", None)
    device = input_dict.get("device", None)
    requires_grad = input_dict.get("requires_grad", False)
    pin_memory = input_dict.get("pin_memory", False)

    data = torch.tensor(data)

    if dtype is not None:
        data = data.to(dtype=dtype)
    
    if not cpu:
        data = data.cuda()
        if device is not None and "cuda" in str(device):
            device = torch.device(device)
    
    if device is not None:
        data = data.to(device)

    data.requires_grad_(requires_grad)
    
    if pin_memory:
        data.pin_memory()

    if not cpu:
        data = data.cpu()

    return {"result": data.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):

    data = input_dict["data"]
    dtype = input_dict.get("dtype", None)
    device = input_dict.get("device", None)
    requires_grad = input_dict.get("requires_grad", False)
    pin_memory = input_dict.get("pin_memory", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        data = tf.constant(data)

        if dtype is not None:
            if dtype == torch.float64:
                data = tf.cast(data, tf.float64)
            elif dtype == torch.float32:
                data = tf.cast(data, tf.float32)
            elif dtype == torch.int64:
                data = tf.cast(data, tf.int64)
            elif dtype == torch.int32:
                data = tf.cast(data, tf.int32)
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
            
        result = data.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "data": np.array([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
        "dtype": torch.float64,
        "requires_grad": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "data": [0, 1]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "data": [[0.11111, 0.222222, 0.3333333]],
        "dtype": torch.float64,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "data": 3.14159
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "data": []
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()