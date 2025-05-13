import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    gather_dim = input_dict["gather_dim"]
    no_split = input_dict.get("no_split", False)

    devices = input_dict["devices"]
    if not cpu:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices = [i % num_gpus for i in devices]  # Ensure device IDs are valid
            input_tensor = input_tensor.cuda(devices[0]) #move to the first gpu
            devices = [torch.device('cuda', i) for i in devices]
        else:
            devices = ['cpu'] * len(devices)
    else:
        devices = ['cpu'] * len(devices)

    result = torch.nn.parallel.scatter_gather.scatter(input_tensor, devices)

    if not cpu:
        result = [r.cpu() for r in result]
        
    numpy_results = [r.numpy() for r in result]

    return {"result": numpy_results}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    gather_dim = input_dict["gather_dim"]
    no_split = input_dict.get("no_split", False)
    
    devices = input_dict["devices"]
    num_devices = len(devices)
    
    input_shape = tf.shape(input_tensor)
    
    
    scattered_tensors = tf.split(input_tensor, num_or_size_splits=num_devices, axis=gather_dim)
    
    numpy_results = [tensor.numpy() for tensor in scattered_tensors]

    return {"result": numpy_results}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
        "gather_dim": 1,
        "devices": [0, 1],
    }
    
    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)
    
    for i in range(len(torch_result["result"])):
      assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
        "gather_dim": 0,
        "devices": [0, 1],
    }
    
    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)
    
    for i in range(len(torch_result["result"])):
      assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
        "gather_dim": 0,
        "devices": [0],
    }
    
    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)
    
    for i in range(len(torch_result["result"])):
      assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()