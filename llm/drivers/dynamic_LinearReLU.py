import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", None)
    if bias is not None:
      bias = torch.tensor(bias)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    model = torch.nn.intrinsic.quantized.dynamic.LinearReLU(input_tensor.shape[1], weight.shape[0])
    model.eval()
    with torch.no_grad():
      model.weight.data.copy_(weight)
      if bias is not None:
          model.bias.data.copy_(bias)

    result = model(input_tensor)

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
        weight = tf.constant(input_dict["weight"])
        bias = input_dict.get("bias", None)
        if bias is not None:
            bias = tf.constant(bias)

        linear = tf.matmul(input_tensor, tf.transpose(weight))
        if bias is not None:
            linear = tf.add(linear, bias)
        relu = tf.nn.relu(linear)

        result = relu.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985], [1.3506, -0.6056]], dtype=np.float32),
        "weight": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        "bias": np.array([0.1, 0.2, 0.3], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.0202, 1.0985], [1.3506, -0.6056]], dtype=np.float32),
        "weight": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()