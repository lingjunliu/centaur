import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    zero_pad = torch.nn.ZeroPad1d(padding)
    result = zero_pad(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    with tf.device(device_string):

        input_tensor_shape = tf.shape(input_tensor)
        rank = tf.rank(input_tensor)

        if isinstance(padding, int):
            padding_before = padding
            padding_after = padding
        elif isinstance(padding, tuple):
            if len(padding) == 2:
                padding_before, padding_after = padding
            else:
                padding_before = padding[0]
                padding_after = padding[1]

        paddings = [[0, 0] for _ in range(rank -1)]
        paddings.append([padding_before, padding_after])
        paddings = tf.constant(paddings)

        result = tf.pad(input_tensor, paddings, "CONSTANT")
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "padding": (2, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "padding": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    input_data = {
        "input": np.array([[1, 2, 3], [4,5,6]], dtype=np.float32),
        "padding": (2, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()