import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"]).unsqueeze(0)
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict["bias_ih"]) if "bias_ih" in input_dict else None
    bias_hh = torch.tensor(input_dict["bias_hh"]) if "bias_hh" in input_dict else None
    hx = torch.tensor(input_dict["hx"]).unsqueeze(0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        if bias_ih is not None:
            bias_ih = bias_ih.cuda()
        if bias_hh is not None:
            bias_hh = bias_hh.cuda()
        hx = hx.cuda()

    if bias_ih is not None and bias_hh is not None:
        output_hx = torch.rnn_relu_cell(input_tensor, hx, weight_ih, weight_hh, bias_ih, bias_hh)
    else:
        output_hx = torch.rnn_relu_cell(input_tensor, hx, weight_ih, weight_hh)

    if not cpu:
        output_hx = output_hx.cpu()

    return {'result': output_hx.numpy(), 'result_h': output_hx.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"][np.newaxis, :])
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias_ih = tf.constant(input_dict["bias_ih"]) if "bias_ih" in input_dict else None
        bias_hh = tf.constant(input_dict["bias_hh"]) if "bias_hh" in input_dict else None
        hx = tf.constant(input_dict["hx"][np.newaxis, :])

        gate_input = tf.matmul(input_tensor, weight_ih)
        gate_hidden = tf.matmul(hx, weight_hh)

        if bias_ih is not None and bias_hh is not None:
            gate = gate_input + gate_hidden + bias_ih + bias_hh
        else:
            gate = gate_input + gate_hidden

        output = tf.nn.relu(gate)

    return {'result': output.numpy(), 'result_h': output.numpy()}


def main():
    A_TOL = 0.01

    input_size = 3
    hidden_size = 2

    input_data = {
        "input": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "weight_ih": np.random.rand(input_size, hidden_size).astype(np.float32),
        "weight_hh": np.random.rand(hidden_size, hidden_size).astype(np.float32),
        "bias_ih": np.random.rand(hidden_size).astype(np.float32),
        "bias_hh": np.random.rand(hidden_size).astype(np.float32),
        "hx": np.array([0.4, 0.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["result_h"], tf_result["result_h"], atol=A_TOL), "Results h do not match"

    print("Success")


if __name__ == "__main__":
    main()