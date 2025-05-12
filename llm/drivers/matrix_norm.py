import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    ord = input_dict.get("ord", 'fro')
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if dim is not None:
        if not isinstance(dim, tuple):
            dim = (dim,)
        dim = tuple(int(d) for d in dim)
    

    result = torch.linalg.matrix_norm(input_tensor, ord=ord, dim=dim, keepdim=keepdim)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        ord_value = input_dict.get("ord", 'fro')
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)

        if ord_value == 'fro':
            ord = 'euclidean'
        elif ord_value == 'nuc':
            s = tf.linalg.svd(input_tensor, compute_uv=False)
            if dim is None:
                result = tf.reduce_sum(s, keepdims=keepdim)
            else:
                result = tf.reduce_sum(s, axis=dim, keepdims=keepdim)
            return {"result": result.numpy()}
        elif ord_value == 1:
            ord = 1
        elif ord_value == -1:
            ord = -1
        elif ord_value == 2:
            s = tf.linalg.svd(input_tensor, compute_uv=False)
            if dim is None:
                result = tf.reduce_max(s, keepdims=keepdim)
            else:
                result = tf.reduce_max(s, axis=dim, keepdims=keepdim)
            return {"result": result.numpy()}
        elif ord_value == -2:
            s = tf.linalg.svd(input_tensor, compute_uv=False)
            if dim is None:
                result = tf.reduce_min(s, keepdims=keepdim)
            else:
                result = tf.reduce_min(s, axis=dim, keepdims=keepdim)
            return {"result": result.numpy()}
        elif ord_value == 'inf':
            ord = np.inf
        elif ord_value == '-inf':
            ord = -np.inf
        else:
            ord = ord_value
        if dim is not None:
            if not isinstance(dim, tuple):
                dim = (dim,)
            dim = tuple(int(d) for d in dim)

        result = tf.norm(input_tensor, ord=ord, axis=dim, keepdims=keepdim)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "ord": 'fro'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "ord": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "ord": np.inf
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "ord": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]],[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        "ord": 'fro',
        "dim": (1,2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]],[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        "ord": 'fro',
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]],[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        "ord": 'nuc',
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()