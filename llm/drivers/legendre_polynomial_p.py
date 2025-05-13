import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    n = torch.tensor(input_dict["n"])
    x = torch.tensor(input_dict["x"])

    if not cpu:
        n = n.cuda()
        x = x.cuda()

    result = torch.special.legendre_polynomial_p(n, x)

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
        n = tf.constant(input_dict["n"])
        x = tf.constant(input_dict["x"])

        def legendre_polynomial_p_tf(n, x):
            n = tf.cast(n, tf.int32)

            def p_0(x):
                return tf.ones_like(x, dtype=tf.float32)
            def p_1(x):
                return tf.identity(x)

            def recursive_calculation(i, p_prev, p_curr, x):
                p_next = ((2.0 * tf.cast(i, tf.float32) - 1.0) * x * p_curr - (tf.cast(i, tf.float32) - 1.0) * p_prev) / tf.cast(i, tf.float32)
                return p_next
            

            max_n = tf.reduce_max(n)
            
            p_values = [p_0(x), p_1(x)]
            
            if max_n >= 2:
                for i in range(2, max_n + 1):
                    p_values.append(recursive_calculation(i, p_values[i-2], p_values[i-1], x))

            result = tf.zeros_like(x, dtype=tf.float32)
            
            num_terms = tf.get_static_value(tf.reduce_max(n)) + 1

            for i in range(num_terms):
                result = tf.where(tf.equal(n, i), p_values[i], result)
            
            return result

        result = legendre_polynomial_p_tf(n, x)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "n": np.array([0, 1, 2, 3], dtype=np.int32),
        "x": np.array([0.2, 0.5, 0.8, -0.1], dtype=np.float32)
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()