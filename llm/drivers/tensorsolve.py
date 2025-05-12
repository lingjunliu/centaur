import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    B = torch.tensor(input_dict["B"])
    dims = input_dict.get("dims", None)

    if not cpu:
        A = A.cuda()
        B = B.cuda()

    X = torch.linalg.tensorsolve(A, B, dims=dims)

    if not cpu:
        X = X.cpu()

    return {"result": X.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    A = tf.constant(input_dict["A"])
    B = tf.constant(input_dict["B"])
    dims = input_dict.get("dims", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        A_shape = A.shape.as_list()
        B_shape = B.shape.as_list()
        B_ndim = len(B_shape)

        if dims is not None:
            dims = list(dims)
            num_dims_moved = len(dims)
            target_positions = list(range(num_dims_moved - len(A_shape) + 1, 1))
            perm = dims + [i for i in range(len(A_shape)) if i not in dims]
            A = tf.transpose(A, perm=perm)
            A_shape = A.shape.as_list()

        m = np.prod(A_shape[:B_ndim])
        n = np.prod(A_shape[B_ndim:])

        A_reshaped = tf.reshape(A, (m, n))
        B_reshaped = tf.reshape(B, (m,))

        try:
            X_reshaped = tf.linalg.solve(A_reshaped, tf.expand_dims(B_reshaped, axis=-1))
            X_shape = A_shape[B_ndim:]
            X = tf.reshape(X_reshaped, X_shape)
        except tf.errors.InvalidArgumentError as e:
             if "is not invertible" in str(e):
                raise RuntimeError("The reshaped A is not invertible") from e
             else:
                raise

    return {"result": X.numpy()}

def main():
    A_TOL = 0.01

    A = np.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4)).astype(np.float32)
    B = np.random.randn(2 * 3, 4).astype(np.float32)
    input_data = {
        "A": A,
        "B": B,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    A = np.random.randn(6, 4, 4, 3, 2).astype(np.float32)
    B = np.random.randn(4, 3, 2).astype(np.float32)
    input_data = {
        "A": A,
        "B": B,
        "dims": (0, 2)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()