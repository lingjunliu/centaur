import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    split_size_or_sections = input_dict["split_size_or_sections"]
    dim = input_dict.get("dim", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.split(input_tensor, split_size_or_sections, dim=dim)

    if not cpu:
        result = [r.cpu() for r in result]

    return {"result": [r.numpy() for r in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    split_size_or_sections = input_dict["split_size_or_sections"]
    dim = input_dict.get("dim", 0)

    if isinstance(split_size_or_sections, int):
        num_splits = (input_tensor.shape[dim] + split_size_or_sections - 1) // split_size_or_sections
        if input_tensor.shape[dim] % split_size_or_sections != 0:
            num_splits = input_tensor.shape[dim] // split_size_or_sections + 1
            split_sizes = [split_size_or_sections] * (input_tensor.shape[dim] // split_size_or_sections)
            remaining = input_tensor.shape[dim] % split_size_or_sections
            if remaining > 0:
              split_sizes.append(remaining)
            split_size_or_sections = split_sizes
        else:
            num_splits = input_tensor.shape[dim] // split_size_or_sections
        
        if isinstance(split_size_or_sections, list):
            result = tf.split(input_tensor, num_or_size_splits=tf.constant(split_size_or_sections, dtype=tf.int32), axis=dim)
        else:
            result = tf.split(input_tensor, num_or_size_splits=num_splits, axis=dim)
    else:
        split_size_or_sections = tf.constant(split_size_or_sections, dtype=tf.int64)
        result = tf.split(input_tensor, split_size_or_sections, axis=dim)

    return {"result": [r.numpy() for r in result]}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
        "split_size_or_sections": [1, 3],
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert len(torch_result["result"]) == len(tf_result["result"]), "Number of splits do not match"
    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
        "split_size_or_sections": 3,
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    
    if isinstance(input_data["split_size_or_sections"], int):
        expected_num_splits = (input_data["input"].shape[0] + input_data["split_size_or_sections"] - 1) // input_data["split_size_or_sections"]
        if input_data["input"].shape[0] % input_data["split_size_or_sections"] != 0:
            expected_num_splits = input_data["input"].shape[0] // input_data["split_size_or_sections"] + 1
            
        assert len(torch_result["result"]) == expected_num_splits, "Number of splits do not match"
        assert len(tf_result["result"]) == expected_num_splits, "Number of splits do not match"
    else:
        assert len(torch_result["result"]) == len(tf_result["result"]), "Number of splits do not match"
    
    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"
        
    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
        "split_size_or_sections": 1,
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert len(torch_result["result"]) == len(tf_result["result"]), "Number of splits do not match"
    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()