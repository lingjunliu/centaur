import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    x = torch.tensor(input_dict["x"])
    n = torch.tensor(input_dict["n"])
    values = torch.tensor(input_dict["values"])

    if not cpu:
        x = x.cuda()
        n = n.cuda()
        values = values.cuda()

    result = torch.special.chebyshev_polynomial_v(x, n, values=values)

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
        x = tf.constant(input_dict["x"])
        n = tf.cast(input_dict["n"], tf.float32)
        values = tf.constant(input_dict["values"])

        def chebyshev_polynomial_v(x, n, values):
            x = tf.cast(x, tf.float32)
            values = tf.cast(values, tf.float32)

            def loop_body(i, prev_val, curr_val):
                next_val = 2 * x * curr_val - prev_val
                return i + 1, curr_val, next_val

            if tf.reduce_any(n < 0.0):
                raise ValueError("n must be non-negative")

            initial_prev = values
            initial_curr = x * values

            def calculate_polynomial(n):
                i = tf.constant(2.0)
                _, _, result = tf.while_loop(
                    lambda i, prev_val, curr_val: i <= n,
                    loop_body,
                    loop_vars=[i, initial_prev, initial_curr],
                    shape_invariants=[i.get_shape(), initial_prev.get_shape(), initial_curr.get_shape()]
                )
                return result

            result = tf.cond(
                tf.equal(n, 0.0),
                lambda: initial_prev,
                lambda: tf.cond(
                    tf.equal(n, 1.0),
                    lambda: initial_curr,
                    lambda: calculate_polynomial(n)
                )
            )
            return result

        result = chebyshev_polynomial_v(x, n, values)
        result = result.numpy()
    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "x": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "n": np.array([2], dtype=np.int32),
        "values": np.array([1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "x": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "n": np.array([3], dtype=np.int32),
        "values": np.array([1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "x": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "n": np.array([0], dtype=np.int32),
        "values": np.array([1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "x": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "n": np.array([1], dtype=np.int32),
        "values": np.array([1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()