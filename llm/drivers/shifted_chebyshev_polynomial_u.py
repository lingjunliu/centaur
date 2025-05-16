import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    n = torch.tensor(input_dict["n"])
    x = torch.tensor(input_dict["x"])

    if not cpu:
        n = n.cuda()
        x = x.cuda()

    result = torch.special.shifted_chebyshev_polynomial_u(n, x)

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
        n = tf.constant(input_dict["n"])
        x = tf.constant(input_dict["x"])

        def shifted_chebyshev_u(n, x):
            if n == 0:
                return tf.ones_like(x)
            elif n == 1:
                return 2 * x - 1
            else:
                u_prev = tf.ones_like(x)
                u_curr = 2 * x - 1
                for i in tf.range(2, n + 1):
                    u_next = 2 * x * u_curr - (u_curr + u_prev)
                    u_prev = u_curr
                    u_curr = u_next
                return u_curr

        if n.shape == ():
            n = tf.cast(n, tf.int32)
            result = shifted_chebyshev_u(n, x)
        else:
            result_list = []
            for i in range(n.shape[0]):
                ni = tf.cast(n[i], tf.int32)
                result_list.append(shifted_chebyshev_u(ni, x))
            result = tf.stack(result_list)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "n": np.array([0, 1, 2, 3], dtype=np.int32),
        "x": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()