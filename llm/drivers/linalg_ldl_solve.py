import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    A = torch.tensor(input_dict["A"])
    b = torch.tensor(input_dict["b"])
    LD = torch.tensor(input_dict["LD"])
    pivots = torch.tensor(input_dict["pivots"]).long() - 1

    if not cpu:
        A = A.cuda()
        b = b.cuda()
        LD = LD.cuda()
        pivots = pivots.cuda()

    result = torch.linalg.ldl_solve(LD, pivots, b, out=None)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    A = tf.constant(input_dict["A"])
    b = tf.constant(input_dict["b"])
    LD = tf.constant(input_dict["LD"])
    pivots = tf.constant(input_dict["pivots"])

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):

        n = A.shape[0]
        x = tf.identity(b)
        pivots = tf.cast(pivots - 1, tf.int32)

        for i in range(n):
            if pivots[i] != i:
                j = pivots[i]
                updates = tf.stack([x[j], x[i]])
                x = tf.tensor_scatter_nd_update(x, [[i], [j]], updates)

        L = tf.linalg.band_part(LD, -1, 0)
        D = tf.linalg.diag(tf.linalg.diag_part(LD))
        U = tf.linalg.band_part(LD, 0, -1) - D + tf.eye(n, dtype=LD.dtype)

        y = tf.linalg.solve(L, tf.expand_dims(x, axis=1))
        z = tf.linalg.solve(D, y)
        result = tf.linalg.solve(tf.transpose(L), z)
        result = tf.reshape(result, [-1])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 4.0], [3.0, 4.0, 9.0]], dtype=np.float32),
        "b": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "LD": np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, -2.0, 1.0]], dtype=np.float32),
        "pivots": np.array([1, 2, 3], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()