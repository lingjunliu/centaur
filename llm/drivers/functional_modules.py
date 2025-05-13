import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.functional as F

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    scale = input_dict.get("scale", 1.0)
    zero_point = input_dict.get("zero_point", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    input_tensor_scaled = (input_tensor.float() - zero_point) * scale
    weight = weight.float()
    if bias is not None:
        bias = bias.float()
    result = F.linear(input_tensor_scaled, weight, bias)

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
        bias = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else None
        scale = input_dict.get("scale", 1.0)
        zero_point = input_dict.get("zero_point", 0)

        input_tensor_scaled = (input_tensor - zero_point) * scale
        result = tf.matmul(input_tensor_scaled, tf.transpose(weight))
        if bias is not None:
            result = tf.add(result, bias)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985], [1.3506, -0.6056]], dtype=np.float32),
        "weight": np.array([[0.5, -0.2], [0.1, 0.8]], dtype=np.float32),
        "bias": np.array([0.1, 0.2], dtype=np.float32),
        "scale": 0.5,
        "zero_point": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()