import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    padding = input_dict["padding"]
    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    layer = torch.nn.ZeroPad2d(padding)

    if not cpu:
        layer = layer.cuda()
    
    result = layer(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    padding = input_dict["padding"]
    input_tensor = tf.constant(input_dict["input"])

    if isinstance(padding, int):
        padding_config = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
    elif isinstance(padding, tuple):
        if len(padding) == 2:
            padding_config = [[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]]
        elif len(padding) == 4:
            padding_config = [[0, 0], [0, 0], [padding[0], padding[1]], [padding[2], padding[3]]]
        else:
            raise ValueError("Padding must be an int, a tuple of length 2, or a tuple of length 4.")
    else:
        raise TypeError("Padding must be an int or tuple.")

    result = tf.pad(input_tensor, padding_config, "CONSTANT")
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[
                [1, 2, 3],
                [4, 5, 6]
            ]]], dtype=np.float32),
        "padding": (1, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_np = torch_result["result"]
    tf_result_np = tf_result["result"]

    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[
                [1, 2, 3],
                [4, 5, 6]
            ]]], dtype=np.float32),
        "padding": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_np = torch_result["result"]
    tf_result_np = tf_result["result"]

    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[
                [1, 2, 3],
                [4, 5, 6]
            ]]], dtype=np.float32),
        "padding": (1, 2, 1, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_np = torch_result["result"]
    tf_result_np = tf_result["result"]

    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()