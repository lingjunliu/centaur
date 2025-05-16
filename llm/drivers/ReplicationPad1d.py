import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    pad1d = torch.nn.ReplicationPad1d(padding)
    
    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor.unsqueeze(0)
        result = pad1d(input_tensor)
        result = result.squeeze(0)
    elif len(input_tensor.shape) == 2:
        result = pad1d(input_tensor)
    else:
        result = pad1d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = tf.shape(input_tensor)
        input_rank = tf.rank(input_tensor)

        if input_rank == 1:
            input_tensor = tf.reshape(input_tensor, [1, -1])
            paddings = [[0, 0], [padding[0], padding[1]]]
            result = tf.pad(input_tensor, paddings, mode='REFLECT')
            result = tf.reshape(result, [-1])
        elif input_rank == 2:
            paddings = [[0, 0], [padding[0], padding[1]]]
            result = tf.pad(input_tensor, paddings, mode='REFLECT', constant_values=0)
        elif input_rank == 3:
            paddings = [[0, 0], [0, 0], [padding[0], padding[1]]]
            result = tf.pad(input_tensor, paddings, mode='REFLECT', constant_values=0)
        else:
            raise ValueError("Tensor rank not supported")

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "padding": (1, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0],[4.0,5.0,6.0]], dtype=np.float32),
        "padding": (1, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[[1.0, 2.0, 3.0],[4.0,5.0,6.0]],[[7.0,8.0,9.0],[10.0,11.0,12.0]]], dtype=np.float32),
        "padding": (1, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()