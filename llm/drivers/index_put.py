import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    indices = input_dict["indices"]
    values = torch.tensor(input_dict["values"])
    accumulate = input_dict.get("accumulate", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        values = values.cuda()
    
    input_tensor_shape = len(input_tensor.shape)
    if input_tensor_shape == 0:
      indices = [torch.tensor(indices)]
    elif input_tensor_shape == 1:
      indices = tuple(torch.tensor(i) for i in zip(*indices))
    else:
      indices = tuple(torch.tensor(i) for i in zip(*indices))

    input_tensor = torch.index_put(input_tensor, indices, values, accumulate=accumulate)

    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        indices = input_dict["indices"]
        values = tf.constant(input_dict["values"])
        accumulate = input_dict.get("accumulate", False)

        input_tensor = tf.identity(input_tensor)
        updates = tf.identity(values)

        if accumulate:
            result = tf.tensor_scatter_nd_add(input_tensor, indices, updates)
        else:
            result = tf.tensor_scatter_nd_update(input_tensor, indices, updates)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "indices": [[0], [2], [4]],
        "values": np.array([7, 8, 9], dtype=np.float32),
        "accumulate": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "indices": [[0], [2], [4]],
        "values": np.array([7, 8, 9], dtype=np.float32),
        "accumulate": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "indices": [[0, 0], [1, 1]],
        "values": np.array([7, 8], dtype=np.float32),
        "accumulate": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "indices": [[0, 0], [1, 1]],
        "values": np.array([7, 8], dtype=np.float32),
        "accumulate": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()