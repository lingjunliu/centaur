import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    hidden_state = torch.tensor(input_dict["hidden_state"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict.get("bias_ih", torch.zeros(4 * input_dict["hidden_size"])))
    bias_hh = torch.tensor(input_dict.get("bias_hh", torch.zeros(4 * input_dict["hidden_size"])))

    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_state = hidden_state.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        bias_ih = bias_ih.cuda()
        bias_hh = bias_hh.cuda()

    lstm_cell = torch.nn.LSTMCell(input_size=input_dict["input_size"], hidden_size=input_dict["hidden_size"])
    lstm_cell.weight_ih = torch.nn.Parameter(weight_ih)
    lstm_cell.weight_hh = torch.nn.Parameter(weight_hh)
    lstm_cell.bias_ih = torch.nn.Parameter(bias_ih)
    lstm_cell.bias_hh = torch.nn.Parameter(bias_hh)
    
    hx, cx = hidden_state[0], hidden_state[1]
    result = lstm_cell(input_tensor, (hx, cx))

    if not cpu:
        result = (result[0].cpu(), result[1].cpu())

    return {"result": (result[0].detach().numpy(), result[1].detach().numpy())}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hidden_state = (tf.constant(input_dict["hidden_state"][0]), tf.constant(input_dict["hidden_state"][1]))
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])

        bias_ih = input_dict.get("bias_ih")
        if bias_ih is None:
            bias_ih = np.zeros(4 * input_dict["hidden_size"])
        bias_ih = tf.constant(bias_ih)

        bias_hh = input_dict.get("bias_hh")
        if bias_hh is None:
            bias_hh = np.zeros(4 * input_dict["hidden_size"])
        bias_hh = tf.constant(bias_hh)


        def lstm_cell(input, hidden_state, weight_ih, weight_hh, bias_ih, bias_hh):
            hx, cx = hidden_state
            
            gates = tf.matmul(input, weight_ih, transpose_b=True) + tf.matmul(hx, weight_hh, transpose_b=True) + bias_ih + bias_hh
            
            input_gate, forget_gate, cell_gate, output_gate = tf.split(gates, num_or_size_splits=4, axis=1)
            
            input_gate = tf.sigmoid(input_gate)
            forget_gate = tf.sigmoid(forget_gate)
            cell_gate = tf.tanh(cell_gate)
            output_gate = tf.sigmoid(output_gate)
            
            cy = forget_gate * cx + input_gate * cell_gate
            hy = output_gate * tf.tanh(cy)

            return hy, cy

        result = lstm_cell(input_tensor, hidden_state, weight_ih, weight_hh, bias_ih, bias_hh)

        return {"result": (result[0].numpy(), result[1].numpy())}

def main():
    A_TOL = 0.01

    input_size = 10
    hidden_size = 20
    batch_size = 1

    input_data = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "input": np.random.randn(batch_size, input_size).astype(np.float32),
        "hidden_state": (np.random.randn(batch_size, hidden_size).astype(np.float32), np.random.randn(batch_size, hidden_size).astype(np.float32)),
        "weight_ih": np.random.randn(4 * hidden_size, input_size).astype(np.float32),
        "weight_hh": np.random.randn(4 * hidden_size, hidden_size).astype(np.float32),
        "bias_ih": np.random.randn(4 * hidden_size).astype(np.float32),
        "bias_hh": np.random.randn(4 * hidden_size).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"][0], tf_result["result"][0], atol=A_TOL), "Output results do not match"
    assert np.allclose(torch_result["result"][1], tf_result["result"][1], atol=A_TOL), "Hidden state results do not match"
    

    print("Success")

if __name__ == "__main__":
    main()