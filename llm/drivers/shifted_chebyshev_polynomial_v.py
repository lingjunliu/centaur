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

    result = torch.special.shifted_chebyshev_polynomial_v(n, x)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    n = tf.constant(input_dict["n"], dtype=tf.int32)
    x = tf.constant(input_dict["x"])

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        def chebyshev_v(n, x):
            n = tf.cast(n, tf.float32)
            if n == 0.0:
                return tf.ones_like(x)
            elif n == 1.0:
                return 2.0 * x - 1.0
            else:
                def loop_body(i, v0, v1):
                    v_next = 2.0 * (2.0 * x - 1.0) * v1 - v0
                    return i + 1, v1, v_next

                i = tf.constant(2)
                v0 = tf.ones_like(x)
                v1 = 2.0 * x - 1.0

                _, _, result = tf.while_loop(
                    lambda i, v0, v1: tf.less_equal(i, tf.cast(n, tf.int32)),
                    loop_body,
                    [i, v0, v1]
                )
                return result

        result = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(tf.shape(n)[0]):
            result = result.write(i, chebyshev_v(tf.cast(n[i],dtype=tf.float32), x))
        result = result.stack()

    return {"result": result.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "n": np.array([0, 1, 2, 3], dtype=np.int32),
        "x": np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()