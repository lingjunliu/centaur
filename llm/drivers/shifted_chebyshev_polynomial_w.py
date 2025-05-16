import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    n = input_dict["n"]
    x = torch.tensor(input_dict["x"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        x = x.cuda()

    result = torch.special.shifted_chebyshev_polynomial_w(x, n, input=input_tensor)

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
        input_tensor = tf.constant(input_dict["input"])
        n = input_dict["n"]
        x = tf.constant(input_dict["x"])
        n = tf.cast(n, dtype=tf.int32)

        def chebyshev_w(x, n, initial_tensor):
            if n == 0:
                return tf.identity(initial_tensor)
            elif n == 1:
                return x * initial_tensor
            else:
                w0 = tf.identity(initial_tensor)
                w1 = x * initial_tensor
                for i in range(2, n + 1):
                    w2 = 2 * x * w1 - w0
                    w0 = w1
                    w1 = w2
                return w1

        result = chebyshev_w(x, n, initial_tensor)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "n": 2,
        "x": np.array(0.5, dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()