import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])
    weight_ih = torch.quantize_per_tensor(torch.tensor(input_dict["weight_ih"]), input_dict["scale_ih"], input_dict["zero_point_ih"], torch.qint8)
    weight_hh = torch.quantize_per_tensor(torch.tensor(input_dict["weight_hh"]), input_dict["scale_hh"], input_dict["zero_point_hh"], torch.qint8)
    bias_ih = torch.tensor(input_dict["bias_ih"], dtype=torch.float)
    bias_hh = torch.tensor(input_dict["bias_hh"], dtype=torch.float)

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        bias_ih = bias_ih.cuda()
        bias_hh = bias_hh.cuda()

    q_input = torch.quantize_per_tensor(input_tensor, input_dict["scale_ih"], input_dict["zero_point_ih"], torch.quint8)
    q_hx = torch.quantize_per_tensor(hx, input_dict["scale_hh"], input_dict["zero_point_hh"], torch.quint8)

    weight_ih_dequant = weight_ih.dequantize()
    weight_hh_dequant = weight_hh.dequantize()

    with torch.no_grad():
        igates = torch.matmul(q_input.dequantize(), weight_ih_dequant.transpose(0, 1)) + bias_ih
        hgates = torch.matmul(q_hx.dequantize(), weight_hh_dequant.transpose(0, 1)) + bias_hh
        output = torch.tanh(igates)

    if not cpu:
        output = output.cpu()

    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hx = tf.constant(input_dict["hx"])
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias_ih = tf.constant(input_dict["bias_ih"])
        bias_hh = tf.constant(input_dict["bias_hh"])

        igates = tf.matmul(input_tensor, tf.transpose(weight_ih)) + bias_ih
        hgates = tf.matmul(hx, tf.transpose(weight_hh)) + bias_hh

        output = tf.tanh(igates + hgates)

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_size = 3
    hidden_size = 2

    input_data = {
        "input": np.random.randn(1, input_size).astype(np.float32),
        "hx": np.random.randn(1, hidden_size).astype(np.float32),
        "weight_ih": np.random.randn(hidden_size, input_size).astype(np.float32),
        "weight_hh": np.random.randn(hidden_size, hidden_size).astype(np.float32),
        "bias_ih": np.random.randn(hidden_size).astype(np.float32),
        "bias_hh": np.random.randn(hidden_size).astype(np.float32),
        "scale_ih": 0.1,
        "scale_hh": 0.2,
        "zero_point_ih": 0,
        "zero_point_hh": 0,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()