import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    pad = torch.nn.ConstantPad3d(padding, value)
    result = pad(input_tensor)

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

        rank = len(input_tensor.shape)
        paddings = [[0, 0]] * rank

        torch_padding = input_dict["padding"]
        tf_padding = [torch_padding[4], torch_padding[5], torch_padding[2], torch_padding[3], torch_padding[0], torch_padding[1]]

        paddings[-4] = [tf_padding[0], tf_padding[1]]
        paddings[-3] = [tf_padding[2], tf_padding[3]]
        paddings[-2] = [tf_padding[4], tf_padding[5]]

        result = tf.pad(input_tensor, paddings, "CONSTANT", constant_values=value)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "padding": (1, 2, 3, 4, 5, 6),
        "value": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()