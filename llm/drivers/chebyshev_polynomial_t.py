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

    result = torch.special.chebyshev_polynomial_t(n, x)

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
        n = tf.constant(input_dict["n"])
        x = tf.constant(input_dict["x"])

        def chebyshev_polynomial_t(n, x):
            n = tf.cast(n, tf.float32)
            if tf.reduce_all(tf.equal(n, 0)):
                return tf.ones_like(x, dtype=tf.float32)
            elif tf.reduce_all(tf.equal(n, 1)):
                return x
            else:
                t0 = tf.ones_like(x, dtype=tf.float32)
                t1 = x
                
                def loop_body(i, t0, t1):
                    t2 = 2 * x * t1 - t0
                    return i + 1, t1, t2
                
                i = tf.constant(2)
                
                _, _, result = tf.while_loop(
                    lambda i, t0, t1: i <= tf.cast(n, tf.int32),
                    loop_body,
                    [i, t0, t1]
                )
                return result

        result = tf.cond(tf.equal(tf.rank(n),0), lambda: chebyshev_polynomial_t(n, x), lambda: tf.map_fn(lambda ni: chebyshev_polynomial_t(ni, x), n))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "n": np.array(3, dtype=np.int32),
        "x": np.array(0.5, dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "n": np.array(np.arange(4), dtype=np.int32),
        "x": np.array(0.5, dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "n": np.array(3, dtype=np.int32),
        "x": np.array(np.arange(4, dtype=np.float32)/3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "n": np.array([3,2,1,0], dtype=np.int32),
        "x": np.array(np.arange(4, dtype=np.float32)/3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()