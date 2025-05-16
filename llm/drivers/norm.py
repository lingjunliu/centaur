import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    ord_val = input_dict.get("ord", None)
    dim_val = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    dtype = input_dict.get("dtype", None)

    if not cpu:
        A = A.cuda()

    if dtype is not None:
        A = A.to(dtype)

    result = torch.linalg.norm(A, ord=ord_val, dim=dim_val, keepdim=keepdim)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    A = tf.constant(input_dict["A"])
    ord_val = input_dict.get("ord", None)
    dim_val = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    dtype = input_dict.get("dtype", None)

    if dtype is not None:
        A = tf.cast(A, dtype=dtype)

    if dim_val is None:
        if ord_val is None:
            A = tf.reshape(A, [-1])
            result = tf.norm(A)
        elif ord_val == 'fro':
            result = tf.norm(A)
        elif ord_val == float('inf'):
            result = tf.reduce_max(tf.abs(A))
        elif ord_val == float('-inf'):
            result = tf.reduce_min(tf.abs(A))
        elif ord_val == 1:
            result = tf.reduce_max(tf.reduce_sum(tf.abs(A), axis=0))
        elif ord_val == -1:
            result = tf.reduce_min(tf.reduce_sum(tf.abs(A), axis=0))
        elif ord_val == 2:
            s = tf.linalg.svd(A, compute_uv=False)
            result = s[0]
        elif ord_val == -2:
            s = tf.linalg.svd(A, compute_uv=False)
            result = s[-1]
        else:
            result = tf.pow(tf.reduce_sum(tf.pow(tf.abs(A), ord_val)), 1/ord_val)
    else:
        if isinstance(dim_val, int):
            if ord_val is None or ord_val == 2:
                result = tf.norm(A, axis=dim_val, keepdims=keepdim)
            elif ord_val == float('inf'):
                result = tf.reduce_max(tf.abs(A), axis=dim_val, keepdims=keepdim)
            elif ord_val == float('-inf'):
                result = tf.reduce_min(tf.abs(A), axis=dim_val, keepdims=keepdim)
            elif ord_val == 1:
                result = tf.reduce_sum(tf.abs(A), axis=dim_val, keepdims=keepdim)
            elif ord_val == -1:
                result = tf.reduce_min(tf.abs(A), axis=dim_val, keepdims=keepdim)
            else:
                result = tf.pow(tf.reduce_sum(tf.pow(tf.abs(A), ord_val), axis=dim_val, keepdims=keepdim), 1/ord_val)
        else:
            if ord_val is None or ord_val == 'fro':
                result = tf.norm(A, ord='euclidean', axis=dim_val, keepdims=keepdim)
            elif ord_val == 1:
                result = tf.reduce_max(tf.reduce_sum(tf.abs(A), axis=dim_val[0]), axis=dim_val[1])
            elif ord_val == -1:
                result = tf.reduce_min(tf.reduce_sum(tf.abs(A), axis=dim_val[0]), axis=dim_val[1])
            else:
                raise ValueError(f"Tensorflow does not support this combination of ord {ord_val} and dim {dim_val}")

    result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1., 2., 3.], [-1, 1, 4]], dtype=np.float32),
        "dim": 1,
        "ord": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "A": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        "dim": (1,2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[-4., -3., -2.], [-1., 0., 1.], [ 2., 3., 4.]], dtype=np.float32),
        "ord": float('inf')
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[-4., -3., -2.], [-1., 0., 1.], [ 2., 3., 4.]], dtype=np.float32),
        "ord": -1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "A": np.arange(9, dtype=np.float32) - 4
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.array([[-4., -3., -2.], [-1., 0., 1.], [ 2., 3., 4.]], dtype=np.float32),
        "ord": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "A": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        "dim": (1,2),
        "ord": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()