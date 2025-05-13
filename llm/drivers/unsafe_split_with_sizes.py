import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    split_size = input_dict["split_size"]
    dim = input_dict.get("dim", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.unsafe_split_with_sizes(input_tensor, split_size, dim=dim)

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
        split_size = input_dict["split_size"]
        dim = input_dict.get("dim", 0)

        input_shape = input_tensor.shape
        dim_size = input_shape[dim]

        split_indices = [0]
        current_index = 0
        for size in split_size:
            current_index += size
            split_indices.append(current_index)
        
        if split_indices[-1] != dim_size:
            raise ValueError("Sum of split sizes does not match dimension size")

        result_list = []
        for i in range(len(split_size)):
            start = split_indices[i]
            end = split_indices[i+1]

            if dim == 0:
                slice_indices = (slice(start, end),) + tuple(slice(None) for _ in range(len(input_shape) - 1))
            elif dim == 1:
                slice_indices = (slice(None), slice(start, end)) + tuple(slice(None) for _ in range(len(input_shape) - 2))
            elif dim == 2:
                slice_indices = (slice(None), slice(None), slice(start, end)) + tuple(slice(None) for _ in range(len(input_shape) - 3))
            elif dim == 3:
                 slice_indices = (slice(None), slice(None), slice(None), slice(start, end)) + tuple(slice(None) for _ in range(len(input_shape) - 4))
            else:
                 slice_indices = (slice(None), slice(None), slice(None), slice(None), slice(start, end)) + tuple(slice(None) for _ in range(len(input_shape) - 5))

            result_list.append(input_tensor[slice_indices].numpy())
            
    return {"result": result_list}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32).reshape(3,4),
        "split_size": [1, 1, 1],
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32).reshape(3,4),
        "split_size": [1, 3],
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()