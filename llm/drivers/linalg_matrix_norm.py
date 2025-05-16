import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    ord_val = input_dict.get("ord", 'fro')
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    dtype = input_dict.get("dtype", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if dim is not None and not isinstance(dim, tuple):
        dim = (dim,)

    if isinstance(ord_val, str):
        result = torch.linalg.matrix_norm(input_tensor, ord=ord_val, dim=dim, keepdim=keepdim, dtype=dtype)
    else:
        result = torch.linalg.matrix_norm(input_tensor, ord=float(ord_val), dim=dim, keepdim=keepdim, dtype=dtype)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        ord_val = input_dict.get("ord", 'fro')
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)
        dtype = input_dict.get("dtype", None)

        if ord_val == 'fro':
            result = tf.norm(input_tensor, ord='euclidean', axis=dim, keepdims=keepdim)
        elif ord_val == 'nuc':
            s = tf.linalg.svd(input_tensor, compute_uv=False)
            result = tf.reduce_sum(s, axis=dim, keepdims=keepdim)
        elif ord_val == 1:
            if dim is None:
                input_tensor = tf.cast(input_tensor, dtype=tf.float32)
                s = tf.linalg.svd(input_tensor, compute_uv=False)
                result = tf.reduce_max(s)
            else:
                result = tf.reduce_max(tf.reduce_sum(tf.abs(input_tensor), axis=dim[0], keepdims=keepdim), axis=None if dim[1] is None else dim[1], keepdims=keepdim)
        elif ord_val == -1:
            if dim is None:
                input_tensor = tf.cast(input_tensor, dtype=tf.float32)
                s = tf.linalg.svd(input_tensor, compute_uv=False)
                result = tf.reduce_min(s)
            else:
                result = tf.reduce_min(tf.reduce_sum(tf.abs(input_tensor), axis=dim[0], keepdims=keepdim), axis=None if dim[1] is None else dim[1], keepdims=keepdim)
        elif ord_val == 2:
            if dim is None:
                input_tensor = tf.cast(input_tensor, dtype=tf.float32)
                s = tf.linalg.svd(input_tensor, compute_uv=False)
                result = s[..., 0]
            else:
                s = tf.linalg.svd(input_tensor, compute_uv=False)
                result = tf.math.sqrt(tf.reduce_sum(tf.math.square(s), axis=dim, keepdims=keepdim))
        elif ord_val == -2:
            input_tensor = tf.cast(input_tensor, dtype=tf.float32)
            s = tf.linalg.svd(input_tensor, compute_uv=False)
            result = tf.reduce_min(s, axis=dim, keepdims=keepdim)
        elif ord_val == 'inf':
            if dim is None:
              result = tf.reduce_max(tf.abs(input_tensor))
            else:
              result = tf.reduce_max(tf.reduce_sum(tf.abs(input_tensor), axis=dim[1], keepdims=keepdim), axis=None if dim[0] is None else dim[0], keepdims=keepdim)
        elif ord_val == '-inf':
            if dim is None:
              result = tf.reduce_min(tf.abs(input_tensor))
            else:
              result = tf.reduce_min(tf.reduce_sum(tf.abs(input_tensor), axis=dim[1], keepdims=keepdim), axis=None if dim[0] is None else dim[0], keepdims=keepdim)
        else:
            raise ValueError(f"Unsupported norm order: {ord_val}")

        if dtype is not None:
            result = tf.cast(result, dtype=dtype)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "ord": 'fro',
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "ord": 1,
        "dim": (0,1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "ord": 'inf',
        "dim": (0,1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "ord": 2,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()