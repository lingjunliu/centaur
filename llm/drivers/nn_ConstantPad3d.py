import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = nn.ConstantPad3d(padding, value)
    output = m(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    if isinstance(padding, int):
        padding = ((0, 0), (0, 0), (padding, padding), (padding, padding), (padding, padding))
    else:
        padding = ((0, 0), (0, 0), (padding[4], padding[5]), (padding[2], padding[3]), (padding[0], padding[1]))
        
    output = tf.pad(input_tensor, padding, constant_values=value)

    return {"result": output.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 1, 5, 5, 5).astype(np.float32),
        "padding": (1, 2, 3, 4, 0, 1),
        "value": 3.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()