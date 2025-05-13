import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    tensors = [torch.tensor(t) for t in input_dict["tensors"]]
    indexing = input_dict.get("indexing", None)

    if not cpu:
        tensors = [t.cuda() for t in tensors]

    result = torch.meshgrid(*tensors, indexing=indexing)

    if not cpu:
        result = [r.cpu() for r in result]

    return {"result": [r.numpy() for r in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    tensors = [tf.convert_to_tensor(t) for t in input_dict["tensors"]]
    indexing = input_dict.get("indexing", None)

    if indexing is None or indexing == 'ij':
        result = tf.meshgrid(*tensors, indexing='ij')
    elif indexing == 'xy':
        result = tf.meshgrid(*tensors, indexing='xy')
    else:
        raise ValueError(f"Invalid indexing mode: {indexing}.  Must be 'ij' or 'xy'.")

    return {"result": [r.numpy() for r in result]}

def main():
    A_TOL = 0.01

    input_data = {
        "tensors": [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32)],
        "indexing": 'ij'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"
    
    input_data = {
        "tensors": [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32)],
        "indexing": 'xy'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()