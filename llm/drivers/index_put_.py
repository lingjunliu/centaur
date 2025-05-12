import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    indices = input_dict["indices"]
    values = torch.tensor(input_dict["values"])
    accumulate = input_dict.get("accumulate", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(indices, tuple):
            indices = tuple(torch.tensor(np.array(idx)).long().cuda() for idx in indices)
        else:
            indices = (torch.tensor(np.array(indices)).long().cuda(),)
        values = values.cuda()
    else:
        if isinstance(indices, tuple):
            indices = tuple(torch.tensor(np.array(idx)).long() for idx in indices)
        else:
            indices = (torch.tensor(np.array(indices)).long(),)

    input_tensor.index_put_(indices, values, accumulate=accumulate)

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

        input_tensor_np = input_tensor.numpy()
        values_np = values.numpy()

        if isinstance(indices, tuple):
            indices_np = tuple(np.array(idx) for idx in indices)
            input_tensor_np[indices_np] = values_np if not accumulate else input_tensor_np[indices_np] + values_np
        else:
            indices_np = np.array(indices)
            input_tensor_np[indices_np] = values_np if not accumulate else input_tensor_np[indices_np] + values_np
            
        result = input_tensor_np

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "indices": [0, 2, 4],
        "values": np.array([7, 8, 9], dtype=np.float32),
        "accumulate": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "indices": ([0, 2], [0, 1]),
        "values": np.array([7, 8], dtype=np.float32),
        "accumulate": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "indices": [0, 2, 4],
        "values": np.array([7, 8, 9], dtype=np.float32),
        "accumulate": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()