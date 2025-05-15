import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0]))) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    weight = weight.unsqueeze(0)

    result = torch.nn.functional.conv_transpose1d(
        input_tensor, 
        weight, 
        bias, 
        stride=stride, 
        padding=padding, 
        output_padding=output_padding, 
        groups=groups, 
        dilation=dilation
    ).squeeze().squeeze()
    
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0])), dtype=tf.float32) if "bias" in input_dict else None
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        
        input_tensor = tf.reshape(input_tensor, (1, input_tensor.shape[0], 1))
        weight = tf.reshape(weight, (1, weight.shape[0], weight.shape[1]))
        weight = tf.transpose(weight, perm=[0, 2, 1])

        output_shape = tf.TensorShape([1, (input_tensor.shape[1] - 1) * stride + weight.shape[1] - 2 * padding + output_padding, 1])
        
        result = tf.nn.conv2d_transpose(
            input_tensor,
            weight,
            output_shape=output_shape,
            strides=[1, stride, 1, 1],
            padding="VALID",
            data_format="NHWC"
        )

        result = tf.reshape(result, (-1,))

        if bias is not None:
            result = result + bias

        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "weight": np.array([[0.5, 0.2], [0.3, 0.1], [0.4, 0.6]], dtype=np.float32),
        "bias": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "groups": 1,
        "dilation": 1,
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()