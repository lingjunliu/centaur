import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    LU_data = torch.tensor(input_dict["LU_data"])
    LU_pivots = torch.tensor(input_dict["LU_pivots"])
    unpack_data = input_dict.get("unpack_data", True)
    unpack_pivots = input_dict.get("unpack_pivots", True)

    if not cpu:
        LU_data = LU_data.cuda()
        LU_pivots = LU_pivots.cuda()

    P, L, U = torch.lu_unpack(LU_data, LU_pivots, unpack_data=unpack_data, unpack_pivots=unpack_pivots)

    if not cpu:
        P = P.cpu()
        L = L.cpu()
        U = U.cpu()

    return {"P": P.numpy(), "L": L.numpy(), "U": U.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    LU_data_np = input_dict["LU_data"]
    LU_pivots_np = input_dict["LU_pivots"]
    unpack_data = input_dict.get("unpack_data", True)
    unpack_pivots = input_dict.get("unpack_pivots", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        LU_data = tf.constant(LU_data_np)
        LU_pivots = tf.constant(LU_pivots_np)

        m = LU_data.shape[-2]
        n = LU_data.shape[-1]

        P = tf.eye(m, dtype=LU_data.dtype)
        L = tf.eye(m, dtype=LU_data.dtype)
        U = tf.zeros([m, n], dtype=LU_data.dtype)

        LU_pivots = tf.cast(LU_pivots, dtype=tf.int32)

        def apply_permutation(matrix, pivots):
            rows = tf.range(matrix.shape[0])
            for i in range(matrix.shape[0]):
              temp = tf.identity(matrix[i])
              matrix = tf.tensor_scatter_nd_update(matrix, [[i]], [matrix[pivots[i]-1]])
              matrix = tf.tensor_scatter_nd_update(matrix, [[pivots[i]-1]], [temp])
            return matrix


        P = apply_permutation(P, LU_pivots)

        triangular_lower_mask = tf.linalg.band_part(tf.ones([m, m], dtype=LU_data.dtype), -1, 0)
        L = tf.multiply(LU_data, triangular_lower_mask)
        L = tf.linalg.set_diag(L, tf.ones(m, dtype=LU_data.dtype))

        triangular_upper_mask = tf.linalg.band_part(tf.ones([m, n], dtype=LU_data.dtype), 0, -1)
        U = tf.multiply(LU_data, triangular_upper_mask)

        P = P.numpy()
        L = L.numpy()
        U = U.numpy()

    return {"P": P, "L": L, "U": U}


def main():
    A_TOL = 0.01

    input_data = {
        "LU_data": np.array([[-1.5092, -1.1362, 0.4960],
                             [0.6288, -0.4072, -0.0503],
                             [-0.6326, 0.3374, 0.7387]], dtype=np.float32),
        "LU_pivots": np.array([1, 3, 3], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["P"], tf_result["P"], atol=A_TOL), "P results do not match"
    assert np.allclose(torch_result["L"], tf_result["L"], atol=A_TOL), "L results do not match"
    assert np.allclose(torch_result["U"], tf_result["U"], atol=A_TOL), "U results do not match"

    print("Success")


if __name__ == "__main__":
    main()