import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    pad3d = torch.nn.ConstantPad3d(padding, value)
    result = pad3d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    pad_width = [[0, 0] for _ in input_tensor.shape.as_list()]

    pad_width[-3][0] = padding[0]
    pad_width[-3][1] = padding[1]
    pad_width[-2][0] = padding[2]
    pad_width[-2][1] = padding[3]
    pad_width[-1][0] = padding[4]
    pad_width[-1][1] = padding[5]

    result = tf.pad(input_tensor, paddings=pad_width, mode='CONSTANT', constant_values=value)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "padding": (1, 1, 2, 2, 3, 3),
        "value": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()