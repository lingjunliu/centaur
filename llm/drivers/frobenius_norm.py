import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if dim is not None:
      if isinstance(dim, int):
        if input_tensor.ndim > 2:
            result = torch.linalg.norm(input_tensor, ord='fro', dim=(dim,), keepdim=keepdim)
        else:
            if input_tensor.ndim == 2:
                result = torch.linalg.norm(input_tensor, ord='fro', dim=(dim,), keepdim=keepdim)
            else:
                result = torch.linalg.norm(input_tensor, ord='fro', dim=(dim,), keepdim=keepdim)

      elif isinstance(dim, list) or isinstance(dim, tuple):
        result = torch.linalg.norm(input_tensor, ord='fro', dim=dim, keepdim=keepdim)
      else:
        raise ValueError("Dim must be an integer, list, or tuple")

    else:
        result = torch.linalg.norm(input_tensor, ord='fro')

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)

        if dim is None:
            result = tf.norm(tf.reshape(input_tensor, [-1]))
            if keepdim:
                result = tf.reshape(result, [1])
        else:
            if isinstance(dim, int):
                result = tf.norm(input_tensor, ord='euclidean', axis=(dim,), keepdims=keepdim)
            elif isinstance(dim, list) or isinstance(dim, tuple):
                result = tf.norm(input_tensor, ord='euclidean', axis=dim, keepdims=keepdim)
            else:
                raise ValueError("Dim must be an integer, list, or tuple")
        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "dim": 0,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        "dim": (0, 1),
        "keepdim": False
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)


    input_data = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        "dim": [0, 1],
        "keepdim": False
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    input_data = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()