import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hidden_state = torch.tensor(input_dict["hx"])
    cell_state = torch.tensor(input_dict["cx"])
    w_ih = torch.tensor(input_dict["w_ih"])
    w_hh = torch.tensor(input_dict["w_hh"])
    b_ih = torch.tensor(input_dict.get("b_ih", np.zeros(4 * input_dict["hidden_size"])).astype(np.float32))
    b_hh = torch.tensor(input_dict.get("b_hh", np.zeros(4 * input_dict["hidden_size"])).astype(np.float32))

    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_state = hidden_state.cuda()
        cell_state = cell_state.cuda()
        w_ih = w_ih.cuda()
        w_hh = w_hh.cuda()
        b_ih = b_ih.cuda()
        b_hh = b_hh.cuda()

    lstm_cell = torch.nn.LSTMCell(input_size=input_dict["input_size"], hidden_size=input_dict["hidden_size"])
    lstm_cell.weight_ih = torch.nn.Parameter(w_ih)
    lstm_cell.weight_hh = torch.nn.Parameter(w_hh)
    lstm_cell.bias_ih = torch.nn.Parameter(b_ih)
    lstm_cell.bias_hh = torch.nn.Parameter(b_hh)
    
    result = lstm_cell(input_tensor, (hidden_state, cell_state))

    if not cpu:
        result = (result[0].cpu(), result[1].cpu())

    return {"result_hx": result[0].detach().numpy(), "result_cx": result[1].detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hidden_state = tf.constant(input_dict["hx"])
        cell_state = tf.constant(input_dict["cx"])
        w_ih = tf.constant(input_dict["w_ih"])
        w_hh = tf.constant(input_dict["w_hh"])
        b_ih = tf.constant(input_dict.get("b_ih", np.zeros(4 * input_dict["hidden_size"])).astype(np.float32))
        b_hh = tf.constant(input_dict.get("b_hh", np.zeros(4 * input_dict["hidden_size"])).astype(np.float32))

        def lstm_cell(input_tensor, hidden_state, cell_state, w_ih, w_hh, b_ih, b_hh):
            
            batch_size = tf.shape(input_tensor)[0]
            
            gate_inputs = tf.matmul(input_tensor, w_ih, transpose_b=True) + tf.matmul(hidden_state, w_hh, transpose_b=True) + b_ih + b_hh

            input_gate, forget_gate, cell_gate, output_gate = tf.split(gate_inputs, num_or_size_splits=4, axis=1)

            input_gate = tf.sigmoid(input_gate)
            forget_gate = tf.sigmoid(forget_gate)
            cell_gate = tf.tanh(cell_gate)
            output_gate = tf.sigmoid(output_gate)

            new_cell_state = forget_gate * cell_state + input_gate * cell_gate
            new_hidden_state = output_gate * tf.tanh(new_cell_state)

            return new_hidden_state, new_cell_state
        
        result_hx, result_cx = lstm_cell(input_tensor, hidden_state, cell_state, w_ih, w_hh, b_ih, b_hh)

    return {"result_hx": result_hx.numpy(), "result_cx": result_cx.numpy()}


def main():
    A_TOL = 0.01

    input_size = 10
    hidden_size = 20
    batch_size = 2

    input_data = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "input": np.random.rand(batch_size, input_size).astype(np.float32),
        "hx": np.random.rand(batch_size, hidden_size).astype(np.float32),
        "cx": np.random.rand(batch_size, hidden_size).astype(np.float32),
        "w_ih": np.random.rand(4 * hidden_size, input_size).astype(np.float32),
        "w_hh": np.random.rand(4 * hidden_size, hidden_size).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result_hx"], tf_result["result_hx"], atol=A_TOL), "Results do not match hx"
    assert np.allclose(torch_result["result_cx"], tf_result["result_cx"], atol=A_TOL), "Results do not match cx"

    print("Success")

if __name__ == "__main__":
    main()