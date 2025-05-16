import numpy as np
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    lcm = input_dict.get("lcm", 1)

    if not isinstance(lcm, torch.Tensor):
        lcm = torch.tensor(lcm, dtype=input_tensor.dtype)

    if not cpu:
        input_tensor = input_tensor.cuda()
        lcm = lcm.cuda()

    result = torch.lcm(input_tensor, lcm)

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
        lcm_val = input_dict.get("lcm", 1)

        if not tf.is_tensor(lcm_val):
            lcm = tf.constant(lcm_val, dtype=input_tensor.dtype)
        else:
            lcm = lcm_val

        a = tf.cast(input_tensor, tf.int64)
        b = tf.cast(lcm, tf.int64)

        def gcd(x, y):
            x = tf.math.abs(x)
            y = tf.math.abs(y)

            def body(x, y):
                return (y, tf.math.floormod(x, y))

            def cond(x, y):
                return tf.math.greater(y, 0)

            _, gcd_val = tf.while_loop(cond, body, (x, y))
            return gcd_val


        gcd_val = gcd(a, b)

        result = tf.math.abs(a * b) // gcd_val

        result = tf.cast(result, input_tensor.dtype).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([12, 18, 24], dtype=np.int64),
        "lcm": 6
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([12, 18, 24], dtype=np.int64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()