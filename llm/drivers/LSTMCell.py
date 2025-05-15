import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])
    cx = torch.tensor(input_dict["cx"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict.get("bias_ih", torch.zeros(4 * input_dict["hidden_size"]).numpy()))
    bias_hh = torch.tensor(input_dict.get("bias_hh", torch.zeros(4 * input_dict["hidden_size"]).numpy()))

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()
        cx = cx.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        bias_ih = bias_ih.cuda()
        bias_hh = bias_hh.cuda()

    lstm_cell = torch.nn.LSTMCell(input_size=input_dict["input_size"], hidden_size=input_dict["hidden_size"])
    lstm_cell.weight_ih = torch.nn.Parameter(weight_ih)
    lstm_cell.weight_hh = torch.nn.Parameter(weight_hh)
    if bias_ih is not None:
      lstm_cell.bias_ih = torch.nn.Parameter(bias_ih)
    if bias_hh is not None:
      lstm_cell.bias_hh = torch.nn.Parameter(bias_hh)
    result_hx, result_cx = lstm_cell(input_tensor, (hx, cx))

    if not cpu:
        result_hx = result_hx.cpu()
        result_cx = result_cx.cpu()

    return {"hx": result_hx.detach().numpy(), "cx": result_cx.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hx = tf.constant(input_dict["hx"])
        cx = tf.constant(input_dict["cx"])
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias_ih = input_dict.get("bias_ih", np.zeros(4 * input_dict["hidden_size"]))
        bias_hh = input_dict.get("bias_hh", np.zeros(4 * input_dict["hidden_size"]))
        bias_ih = tf.constant(bias_ih)
        bias_hh = tf.constant(bias_hh)

        input_size = input_dict["input_size"]
        hidden_size = input_dict["hidden_size"]

        gate_inputs = tf.matmul(input_tensor, weight_ih, transpose_b=True) + bias_ih
        recurrent_gate_inputs = tf.matmul(hx, weight_hh, transpose_b=True) + bias_hh

        i, j, f, o = tf.split(gate_inputs + recurrent_gate_inputs, num_or_size_splits=4, axis=1)

        new_c = (cx * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        result_hx = new_h.numpy()
        result_cx = new_c.numpy()

    return {"hx": result_hx, "cx": result_cx}


def main():
    A_TOL = 0.01

    input_size = 3
    hidden_size = 2

    input_data = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "input": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "hx": np.array([[0.4, 0.5]], dtype=np.float32),
        "cx": np.array([[0.6, 0.7]], dtype=np.float32),
        "weight_ih": np.random.rand(4 * hidden_size, input_size).astype(np.float32),
        "weight_hh": np.random.rand(4 * hidden_size, hidden_size).astype(np.float32),
        "bias_ih": np.random.rand(4 * hidden_size).astype(np.float32),
        "bias_hh": np.random.rand(4 * hidden_size).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["hx"], tf_result["hx"], atol=A_TOL), "hx Results do not match"
    assert np.allclose(torch_result["cx"], tf_result["cx"], atol=A_TOL), "cx Results do not match"

    print("Success")


if __name__ == "__main__":
    main()