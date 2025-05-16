import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])
    w_ih = torch.tensor(input_dict["w_ih"])
    w_hh = torch.tensor(input_dict["w_hh"])
    b_ih = torch.tensor(input_dict.get("b_ih", np.zeros(3 * input_dict["hidden_size"], dtype=np.float32)))
    b_hh = torch.tensor(input_dict.get("b_hh", np.zeros(3 * input_dict["hidden_size"], dtype=np.float32)))
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()
        w_ih = w_ih.cuda()
        w_hh = w_hh.cuda()
        b_ih = b_ih.cuda()
        b_hh = b_hh.cuda()

    gru_cell = torch.nn.quantized.dynamic.modules.GRUCell(input_size=input_dict["input_size"], hidden_size=input_dict["hidden_size"])

    gru_cell.weight_ih = torch.nn.Parameter(w_ih)
    gru_cell.weight_hh = torch.nn.Parameter(w_hh)
    gru_cell.bias_ih = torch.nn.Parameter(b_ih)
    gru_cell.bias_hh = torch.nn.Parameter(b_hh)

    result = gru_cell(input_tensor, hx)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        hx = tf.constant(input_dict["hx"], dtype=tf.float32)
        w_ih = tf.constant(input_dict["w_ih"], dtype=tf.float32)
        w_hh = tf.constant(input_dict["w_hh"], dtype=tf.float32)
        b_ih = tf.constant(input_dict.get("b_ih", np.zeros(3 * input_dict["hidden_size"], dtype=np.float32)), dtype=tf.float32)
        b_hh = tf.constant(input_dict.get("b_hh", np.zeros(3 * input_dict["hidden_size"], dtype=np.float32)), dtype=tf.float32)

        input_size = input_dict["input_size"]
        hidden_size = input_dict["hidden_size"]

        w_ir, w_iz, w_in = tf.split(w_ih, num_or_size_splits=3, axis=0)
        w_hr, w_hz, w_hn = tf.split(w_hh, num_or_size_splits=3, axis=0)
        b_ir, b_iz, b_in = tf.split(b_ih, num_or_size_splits=3, axis=0)
        b_hr, b_hz, b_hn = tf.split(b_hh, num_or_size_splits=3, axis=0)

        r = tf.sigmoid(tf.matmul(input_tensor, tf.transpose(w_ir)) + tf.matmul(hx, tf.transpose(w_hr)) + b_ir + b_hr)
        z = tf.sigmoid(tf.matmul(input_tensor, tf.transpose(w_iz)) + tf.matmul(hx, tf.transpose(w_hz)) + b_iz + b_hz)
        n = tf.tanh(tf.matmul(input_tensor, tf.transpose(w_in)) + tf.matmul(r * hx, tf.transpose(w_hn)) + b_in + b_hn)

        h_next = (1 - z) * n + z * hx

        result = h_next.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_size = 10
    hidden_size = 20

    input_data = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "input": np.random.rand(1, input_size).astype(np.float32),
        "hx": np.random.rand(1, hidden_size).astype(np.float32),
        "w_ih": np.random.rand(3 * hidden_size, input_size).astype(np.float32),
        "w_hh": np.random.rand(3 * hidden_size, hidden_size).astype(np.float32),
        "b_ih": np.random.rand(3 * hidden_size).astype(np.float32),
        "b_hh": np.random.rand(3 * hidden_size).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_result_result = torch_result["result"]
    tf_result_result = tf_result["result"]

    assert np.allclose(torch_result_result, tf_result_result, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()