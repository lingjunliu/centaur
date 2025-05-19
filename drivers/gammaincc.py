import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    a = torch.tensor(input_dict["a"])
    x = torch.tensor(input_dict["x"])

    if not cpu:
        a = a.cuda()
        x = x.cuda()
    
    result = torch.special.gammaincc(a, x)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.math import lgamma, exp, reciprocal, multiply, subtract, add

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        a = tf.constant(input_dict["a"], dtype=tf.float32)
        x = tf.constant(input_dict["x"], dtype=tf.float32)

        def gammaincc(a, x):
            igamma = tf.math.igamma(a, x)
            return 1.0 - igamma

        result = gammaincc(a, x)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "a": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        "x": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()