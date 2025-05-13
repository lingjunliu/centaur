import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    n = torch.tensor(input_dict["n"])
    x = torch.tensor(input_dict["x"])

    if not cpu:
        n = n.cuda()
        x = x.cuda()

    result = torch.special.chebyshev_polynomial_u(n, x)

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
        n = tf.constant(input_dict["n"], dtype=tf.float32)
        x = tf.constant(input_dict["x"], dtype=tf.float32)

        def chebyshev_u(n, x):
            n = tf.cast(n, tf.int32)
            u_0 = tf.ones_like(x)
            u_1 = 2 * x

            def loop(i, u_prev, u_curr):
                u_next = 2 * x * u_curr - u_prev
                return i + 1, u_curr, u_next

            i = tf.constant(2)
            u_prev = u_0
            u_curr = u_1
            
            if n == 0:
                return u_0
            elif n == 1:
                return u_1
            else:

                _, u_prev, u_curr = tf.while_loop(
                    lambda i, u_prev, u_curr: i <= n,
                    loop,
                    (i, u_prev, u_curr)
                )

                return u_curr

        result = chebyshev_u(n, x)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "n": np.array(3, dtype=np.float32),
        "x": np.array(0.5, dtype=np.float32)
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "n": np.array([0, 1, 2, 3], dtype=np.float32),
        "x": np.array(0.5, dtype=np.float32)
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "n": np.array(3, dtype=np.float32),
        "x": np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float32)
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    input_data = {
        "n": np.array([0, 1, 2, 3], dtype=np.float32),
        "x": np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float32)
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()