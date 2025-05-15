import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    pad1d = torch.nn.ConstantPad1d(padding, value)
    result = pad1d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        padding = input_dict["padding"]
        value = input_dict.get("value", 0.0)

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=2)

        pad_before = padding[0]
        pad_after = padding[1]

        paddings = [[0, 0], [pad_before, pad_after], [0, 0]]
        result = tf.pad(input_tensor, paddings, constant_values=value)

        result = tf.squeeze(result, axis=[0, 2])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "padding": (2, 3),
        "value": 1.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()