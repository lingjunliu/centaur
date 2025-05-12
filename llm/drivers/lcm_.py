import numpy as np
import functools

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    value = torch.tensor(input_dict["value"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        value = value.cuda()

    result = torch.lcm(input_tensor, value)

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
        input_tensor = tf.constant(input_dict["input"])
        value = tf.constant(input_dict["value"])

        def gcd(x, y):
            x = tf.cast(x, tf.int64)
            y = tf.cast(y, tf.int64)
            while tf.reduce_any(tf.not_equal(y, 0)):
                x, y = y, tf.math.floormod(x, y)
            return x
        
        def lcm(x, y):
            x = tf.cast(x, tf.int64)
            y = tf.cast(y, tf.int64)
            return tf.math.abs(x * y) // gcd(x, y)
        
        result = lcm(input_tensor, value)
        result = tf.cast(result, input_tensor.dtype).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([2, 4, 6, 8], dtype=np.int32),
        "value": np.array([3, 5, 7, 9], dtype=np.int32),
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()