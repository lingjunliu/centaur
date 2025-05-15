import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight_tensor = torch.tensor(input_dict["weight"])
    bias_tensor = torch.tensor(input_dict.get("bias", None)) if input_dict.get("bias", None) is not None else None
    padding = input_dict.get("padding", (0, 0))
    stride = input_dict.get("stride", (1, 1))
    dilation = input_dict.get("dilation", (1, 1))
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = weight_tensor.cuda()
        if bias_tensor is not None:
            bias_tensor = bias_tensor.cuda()

    result = torch.nn.functional.conv2d(
        input_tensor.unsqueeze(0).unsqueeze(0),
        weight_tensor.unsqueeze(0).unsqueeze(0),
        bias_tensor,
        stride,
        padding,
        dilation,
        groups
    )

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        weight_tensor = tf.constant(input_dict["weight"])
        bias_tensor = tf.constant(input_dict["bias"]) if input_dict.get("bias", None) is not None else None
        padding = input_dict.get("padding", (0, 0))
        stride = input_dict.get("stride", (1, 1))
        dilation = input_dict.get("dilation", (1, 1))
        groups = input_dict.get("groups", 1)

        input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[0], input_tensor.shape[1], 1])
        weight_tensor = tf.reshape(weight_tensor, [weight_tensor.shape[0], weight_tensor.shape[1], 1, 1])

        padding_tf = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]

        input_padded = tf.pad(input_tensor, padding_tf, "CONSTANT")
        
        stride_tf = [1, stride[0], stride[1], 1]
        dilation_tf = [1, dilation[0], dilation[1], 1]
        
        result = tf.nn.conv2d(input_padded, weight_tensor, strides=stride_tf, padding='VALID', dilations = dilation_tf)
        if bias_tensor is not None:
          result = tf.nn.bias_add(result, bias_tensor)

        result = tf.reshape(result, [result.shape[1], result.shape[2]])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "weight": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        "bias": np.array([1], dtype=np.float32),
        "padding": (0, 0),
        "stride": (1, 1),
        "dilation": (1, 1),
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()