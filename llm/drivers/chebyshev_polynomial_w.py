import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["x"])
    n = torch.tensor(np.array([input_dict["n"]]), dtype=torch.int32)
    normalized = input_dict.get("normalized", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        n = n.cuda()

    result = torch.special.chebyshev_polynomial_w(input_tensor, n, normalized=normalized)

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
        x = tf.constant(input_dict["x"], dtype=tf.float32)
        n = input_dict["n"]
        normalized = input_dict.get("normalized", False)

        def chebyshev_w(x, n, normalized):
            n = tf.cast(n, tf.int32)
            if normalized:
                if n == 0:
                    return tf.ones_like(x, dtype=tf.float32) / tf.constant(np.sqrt(np.pi), dtype=tf.float32)
                elif n == 1:
                    return x * tf.constant(np.sqrt(2 / np.pi), dtype=tf.float32)
                else:
                    w0 = tf.ones_like(x, dtype=tf.float32) / tf.constant(np.sqrt(np.pi), dtype=tf.float32)
                    w1 = x * tf.constant(np.sqrt(2 / np.pi), dtype=tf.float32)
                    for i in range(2, n + 1):
                        w2 = 2 * x * w1 - w0
                        w0 = w1
                        w1 = w2
                    return w1
            else:
                if n == 0:
                    return tf.ones_like(x, dtype=tf.float32)
                elif n == 1:
                    return x
                else:
                    w0 = tf.ones_like(x, dtype=tf.float32)
                    w1 = x
                    for i in range(2, n + 1):
                        w2 = 2 * x * w1 - w0
                        w0 = w1
                        w1 = w2
                    return w1
        result = chebyshev_w(x, n, normalized)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "x": np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
        "n": 2,
        "normalized": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "x": np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
        "n": 2,
        "normalized": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()