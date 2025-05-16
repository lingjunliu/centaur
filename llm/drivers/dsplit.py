import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices_or_sections = input_dict["indices_or_sections"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.dsplit(input_tensor, indices_or_sections)

    if not cpu:
        result = tuple(r.cpu() for r in result)

    return {"result": tuple(r.numpy() for r in result)}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    indices_or_sections = input_dict["indices_or_sections"]

    if isinstance(indices_or_sections, int):
        num_or_size_splits = indices_or_sections
    elif isinstance(indices_or_sections, (list, tuple)):
        num_or_size_splits = []
        prev = 0
        for i in indices_or_sections:
            num_or_size_splits.append(i - prev)
            prev = i

        if input_tensor.shape[2] > indices_or_sections[-1]:
           num_or_size_splits.append(input_tensor.shape[2] - indices_or_sections[-1])
    else:
        raise TypeError("indices_or_sections must be an integer or a list/tuple of integers")

    result = tf.split(input_tensor, num_or_size_splits, axis=2)

    return {"result": tuple(r.numpy() for r in result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(16.0).reshape(2, 2, 4),
        "indices_or_sections": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    min_len = min(len(torch_result["result"]), len(tf_result["result"]))

    for i in range(min_len):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.arange(16.0).reshape(2, 2, 4),
        "indices_or_sections": [3, 4]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    min_len = min(len(torch_result["result"]), len(tf_result["result"]))

    for i in range(min_len):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()