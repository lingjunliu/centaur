import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict.get("padding", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    m = nn.ZeroPad3d(padding)
    result = m(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = input_dict["input"]
    padding = input_dict.get("padding", 0)

    if isinstance(padding, int):
        padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = padding
    else:
        padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back = padding

    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    
    rank = len(input_tensor.shape)
    if rank == 4:
        paddings = [[0, 0], [padding_front, padding_back], [padding_top, padding_bottom], [padding_left, padding_right]]
    elif rank == 5:
        paddings = [[0, 0], [0, 0], [padding_front, padding_back], [padding_top, padding_bottom], [padding_left, padding_right]]
    else:
        raise ValueError("Input tensor must have rank 4 or 5")

    result = tf.pad(input_tensor, paddings, "CONSTANT")

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "padding": (1, 1, 2, 2, 3, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(3, 4, 5, 6).astype(np.float32),
        "padding": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()