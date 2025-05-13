import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    indices_or_sections = input_dict["indices_or_sections"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.vsplit(input_tensor, indices_or_sections)

    if not cpu:
        result = [r.cpu() for r in result]

    return {"result": [r.numpy() for r in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    indices_or_sections = input_dict["indices_or_sections"]

    if isinstance(indices_or_sections, int):
        num_splits = indices_or_sections
        splits = tf.split(input_tensor, num_splits, axis=0)
        result = [s.numpy() for s in splits]
    else:
        split_indices = indices_or_sections
        splits = []
        last_index = 0
        for i in split_indices:
            splits.append(input_tensor[last_index:i])
            last_index = i
        splits.append(input_tensor[last_index:])
        result = [s.numpy() for s in splits]
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data_int = {
        "input": np.arange(16.0).reshape(4, 4),
        "indices_or_sections": 2
    }

    torch_result_int = torch_version(input_data_int)
    tf_result_int = tensorflow_version(input_data_int)

    for i in range(len(torch_result_int["result"])):
        assert np.allclose(torch_result_int["result"][i], tf_result_int["result"][i], atol=A_TOL), "Results do not match (int split)"

    input_data_list = {
        "input": np.arange(16.0).reshape(4, 4),
        "indices_or_sections": [3, 6]
    }

    torch_result_list = torch_version(input_data_list)
    tf_result_list = tensorflow_version(input_data_list)

    for i in range(len(torch_result_list["result"])):
        assert np.allclose(torch_result_list["result"][i], tf_result_list["result"][i], atol=A_TOL), "Results do not match (list split)"

    print("Success")

if __name__ == "__main__":
    main()