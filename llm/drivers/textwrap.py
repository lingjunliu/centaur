import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.functional as F

    input_tensor = torch.tensor(input_dict["input"])
    max_length = input_dict.get("max_length", None)
    fill_value = input_dict.get("fill_value", 0)
    truncate = input_dict.get("truncate", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if max_length is not None:
        padding_size = max(0, max_length - input_tensor.size(0))
        result = F.pad(input_tensor, (0, padding_size), value=fill_value)

        if truncate:
            result = result[:max_length]
    else:
        result = input_tensor

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    max_length = input_dict.get("max_length", None)
    fill_value = input_dict.get("fill_value", 0)
    truncate = input_dict.get("truncate", False)

    input_shape = tf.shape(input_tensor)[0]

    if max_length is not None:
        padding_size = tf.maximum(0, max_length - input_shape)
        padding = [[0, padding_size.numpy()]]
        padding = tf.constant(padding, dtype=tf.int32)

        padded_tensor = tf.pad(input_tensor, padding, constant_values=fill_value)

        if truncate:
            truncated_tensor = padded_tensor[:max_length]
            result = truncated_tensor.numpy()
        else:
            result = padded_tensor.numpy()
    else:
        result = input_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "max_length": 5,
        "fill_value": 0,
        "truncate": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "max_length": 2,
        "fill_value": 0,
        "truncate": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()