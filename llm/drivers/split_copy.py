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

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        split_size_or_sections = input_dict["split_size_or_sections"]
        dim = input_dict.get("dim", 0)

        if isinstance(split_size_or_sections, int):
            num_splits = (input_tensor.shape[dim] + split_size_or_sections - 1) // split_size_or_sections
            split_size = [split_size_or_sections] * (num_splits - 1)
            split_size.append(input_tensor.shape[dim] - sum(split_size))
            split_size = tf.constant(split_size, dtype=tf.int32)
            result = tf.split(input_tensor, split_size, axis=dim)

        elif isinstance(split_size_or_sections, list):
            split_size_or_sections = tf.constant(split_size_or_sections, dtype=tf.int32)
            result = tf.split(input_tensor, split_size_or_sections, axis=dim)
        else:
            raise TypeError("split_size_or_sections must be an integer or a list")

        result = [r.numpy() for r in result]

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "split_size_or_sections": 2,
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for torch_arr, tf_arr in zip(torch_result["result"], tf_result["result"]):
        assert np.allclose(torch_arr, tf_arr, atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        "split_size_or_sections": [1, 1],
        "dim": 0
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for torch_arr, tf_arr in zip(torch_result["result"], tf_result["result"]):
        assert np.allclose(torch_arr, tf_arr, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()