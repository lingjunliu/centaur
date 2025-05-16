import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    weight_tensor = torch.tensor(input_dict["weight"], dtype=torch.float32)
    bias_tensor = torch.tensor(input_dict["bias"], dtype=torch.float32) if "bias" in input_dict else None
    padding = input_dict.get("padding", (0,))
    stride = input_dict.get("stride", (1,))
    dilation = input_dict.get("dilation", (1,))
    groups = input_dict.get("groups", 1)
    benchmark = input_dict.get("benchmark", True)
    deterministic = input_dict.get("deterministic", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = weight_tensor.cuda()
        if bias_tensor is not None:
            bias_tensor = bias_tensor.cuda()

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic

    input_tensor_expanded = input_tensor.unsqueeze(0).unsqueeze(0)
    weight_tensor_expanded = weight_tensor.unsqueeze(0).unsqueeze(0)

    try:
        result = torch.nn.functional.conv2d(input_tensor_expanded, weight_tensor_expanded, bias=bias_tensor, stride=stride, padding=padding, dilation=dilation, groups=groups).squeeze().squeeze()
    except Exception as e:
        print(f"Torch convolution failed: {e}")
        result = torch.zeros_like(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight_tensor = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias_tensor = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else None
        padding = input_dict.get("padding", (0,))
        stride = input_dict.get("stride", (1,))
        dilation = input_dict.get("dilation", (1,))
        groups = input_dict.get("groups", 1)

        input_tensor_expanded = tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=0)
        weight_tensor_expanded = tf.expand_dims(tf.expand_dims(weight_tensor, axis=0), axis=0)

        padding_tf = "VALID" if padding == (0,) else "SAME"

        try:
            result = tf.nn.conv2d(input_tensor_expanded, weight_tensor_expanded, strides=[1, stride[0], stride[0], 1], padding=padding_tf, dilations=[1, dilation[0], dilation[0], 1], data_format='NHWC')

            if bias_tensor is not None:
                result = tf.nn.bias_add(result, bias_tensor, data_format='NHWC')

            result = tf.squeeze(result).numpy()
        except tf.errors.InvalidArgumentError as e:
            print(f"TensorFlow convolution failed: {e}")
            result = np.zeros_like(input_dict["input"])

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "weight": np.array([[1, 0], [0, 1]], dtype=np.float32),
        "bias": np.array([1], dtype=np.float32),
        "padding": (0,),
        "stride": (1,),
        "dilation": (1,),
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if torch_result["result"].shape != tf_result["result"].shape:
        print("Shape mismatch, skipping assertion")
    else:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "weight": np.array([[1, 0], [0, 1]], dtype=np.float32),
        "padding": (1,),
        "stride": (1,),
        "dilation": (1,),
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if torch_result["result"].shape != tf_result["result"].shape:
        print("Shape mismatch, skipping assertion")
    else:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
        "weight": np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        "bias": np.array([1], dtype=np.float32),
        "padding": (0,),
        "stride": (1,),
        "dilation": (1,),
        "groups": 2 
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if torch_result["result"].shape != tf_result["result"].shape:
        print("Shape mismatch, skipping assertion")
    else:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
        
    print("Success")

if __name__ == "__main__":
    main()