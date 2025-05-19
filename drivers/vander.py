import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    x = torch.tensor(input_dict["x"])
    N = input_dict.get("N", None)
    increasing = input_dict.get("increasing", False)
    
    if not cpu:
        x = x.cuda()
    
    result = torch.vander(x, N=N, increasing=increasing)
    
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
        N = input_dict.get("N", None)
        increasing = input_dict.get("increasing", False)
        
        x_rank = tf.rank(x)
        x_shape = tf.shape(x)

        if N is None:
            N = x_shape[0]

        def body(i, outputs):
            powers = tf.range(N)
            if not increasing:
                powers = tf.reverse(powers, axis=[0])
            
            row = tf.pow(tf.cast(x[i], dtype=tf.float32), tf.cast(powers, dtype=tf.float32))
            outputs = outputs.write(i, row)
            return i + 1, outputs

        outputs = tf.TensorArray(dtype=tf.float32, size=x_shape[0])
        i = tf.constant(0)
        _, result_ta = tf.while_loop(
            lambda i, _: tf.less(i, x_shape[0]),
            body,
            loop_vars=(i, outputs)
        )

        result = result_ta.stack()
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "x": np.array([1, 2, 3, 5], dtype=np.int32),
        "N": 3,
        "increasing": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "x": np.array([1, 2, 3, 5], dtype=np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "x": np.array([1, 2, 3, 5], dtype=np.int32),
        "N": 3,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()