import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    split_size_or_sections = input_dict["split_size_or_sections"]
    dim = input_dict.get("dim", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if isinstance(split_size_or_sections, list):
        result = torch.split(input_tensor, split_size_or_sections, dim=dim)
    else:
        result = torch.split(input_tensor, split_size_or_sections, dim=dim)

    if not cpu:
        result = [r.cpu() for r in result]

    return {"result": [r.numpy() for r in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        split_size_or_sections = input_dict["split_size_or_sections"]
        dim = input_dict.get("dim", 0)

        if isinstance(split_size_or_sections, int):
            result = tf.split(input_tensor, num_or_size_splits=split_size_or_sections, axis=dim)
        else:
            result = tf.split(input_tensor, split_size_or_sections, axis=dim)
            
        result = [r.numpy() for r in result]
            
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        "split_size_or_sections": [1, 1, 1, 1],
        "dim": 1
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
      assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        "split_size_or_sections": 2,
        "dim": 1
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
      assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        "split_size_or_sections": 1,
        "dim": 0
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
      assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        "split_size_or_sections": [1,1],
        "dim": 0
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
      assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()