import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    v = torch.tensor(input_dict["v"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        v = v.cuda()
    
    result = torch.linalg.householder_product(input_tensor, v)

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
        input_tensor = tf.cast(tf.constant(input_dict["input"]), dtype=tf.float32)
        v = tf.cast(tf.constant(input_dict["v"]), dtype=tf.float32)

        v_norm_squared = tf.reduce_sum(tf.square(v))
        
        axes = list(range(len(input_tensor.shape) - 1))
        axes.append(len(input_tensor.shape) - 1)

        result = input_tensor - 2 * tf.tensordot(v, tf.tensordot(v, input_tensor, axes=[[0], [len(input_tensor.shape)-1]]), axes=[[0], [0]]) / v_norm_squared
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "v": np.array([0.5, 0.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        "v": np.array([0.5, 0.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()