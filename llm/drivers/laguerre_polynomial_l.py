import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    n = torch.tensor(input_dict["n"])
    x = torch.tensor(input_dict["x"])
    alpha = input_dict.get("alpha", 0.0)

    if not cpu:
        n = n.cuda()
        x = x.cuda()

    result = torch.special.laguerre_polynomial_l(x, n.int())

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
        n = tf.convert_to_tensor(input_dict["n"])
        x = tf.convert_to_tensor(input_dict["x"])
        alpha = input_dict.get("alpha", 0.0)
        alpha = tf.convert_to_tensor(alpha, dtype=x.dtype)

        def laguerre_polynomial(n, alpha, x):
            n = tf.cast(n, dtype=tf.int32)
            if n == 0:
                return tf.ones_like(x)
            elif n == 1:
                return 1 + alpha - x
            else:
                L_n_minus_1 = laguerre_polynomial(n - 1, alpha, x)
                L_n_minus_2 = laguerre_polynomial(n - 2, alpha, x)
                numerator = (2 * (n - 1) + 1 + alpha - x) * L_n_minus_1 - (n - 1 + alpha) * L_n_minus_2
                denominator = n
                return numerator / tf.cast(denominator, dtype=x.dtype)

        if tf.reduce_sum(tf.cast(n >= 0, dtype=tf.int32)) != tf.size(n):
            raise ValueError("n must be non-negative")
            
        if n.shape == ():
            result = laguerre_polynomial(n, alpha, x)
        else:
            result = tf.vectorized_map(lambda ni: laguerre_polynomial(ni, alpha, x), n)

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "n": np.array([0, 1, 2, 3], dtype=np.int32),
        "x": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "alpha": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()