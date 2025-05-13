import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight_tensor = tuple(torch.tensor(w) for w in input_dict["weight"])
    hx = torch.tensor(input_dict["hx"])
    seq_lengths = torch.tensor(input_dict["seq_lengths"])
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    train = input_dict.get("train", True)
    bidirectional = input_dict.get("bidirectional", False)
    batch_sizes = torch.tensor(input_dict["batch_sizes"])
    mode = input_dict.get("mode", 0)
    hidden_size = input_dict["hx"].shape[-1]

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = tuple(w.cuda() for w in weight_tensor)
        hx = hx.cuda()
        seq_lengths = seq_lengths.cuda()
        batch_sizes = batch_sizes.cuda()

    result = torch.miopen_rnn(input_tensor, weight_tensor, int(input_dict["weight"][0].shape[0]), hx, seq_lengths, mode, hidden_size, batch_first, dropout, train, bidirectional, batch_sizes)

    if not cpu:
        result = (result[0].cpu(), result[1].cpu())

    return {"result0": result[0].numpy(), "result1": result[1].numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        weight_tensor = [tf.constant(w) for w in input_dict["weight"]]
        hx = tf.constant(input_dict["hx"])
        seq_lengths = tf.constant(input_dict["seq_lengths"])
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        train = input_dict.get("train", True)
        bidirectional = input_dict.get("bidirectional", False)
        batch_sizes = tf.constant(input_dict["batch_sizes"])
        
        hidden_size = input_dict["hx"].shape[-1]

        def lstm_cell(input_tensor, state, weights):
            W_i, W_f, W_c, W_o, R_i, R_f, R_c, R_o, b_i, b_f, b_c, b_o = weights
            h_prev, c_prev = state
            
            i = tf.sigmoid(tf.matmul(input_tensor, W_i) + tf.matmul(h_prev, R_i) + b_i)
            f = tf.sigmoid(tf.matmul(input_tensor, W_f) + tf.matmul(h_prev, R_f) + b_f)
            c_tilda = tf.tanh(tf.matmul(input_tensor, W_c) + tf.matmul(h_prev, R_c) + b_c)
            o = tf.sigmoid(tf.matmul(input_tensor, W_o) + tf.matmul(h_prev, R_o) + b_o)
            
            c = f * c_prev + i * c_tilda
            h = o * tf.tanh(c)
            
            return h, c

        def process_sequence(input_sequence, initial_state, weights):
            states = []
            state = initial_state
            for i in range(input_sequence.shape[0]):
                state = lstm_cell(input_sequence[i], state, weights)
                states.append(state)
            
            h_states, c_states = zip(*states)
            
            h_states = tf.stack(h_states)
            c_states = tf.stack(c_states)

            return h_states, c_states

        num_directions = 2 if bidirectional else 1
        num_layers = 1

        W_i, W_f, W_c, W_o, R_i, R_f, R_c, R_o, b_i, b_f, b_c, b_o = weight_tensor

        h_0 = hx[0]
        c_0 = hx[1]

        h_states, c_states = process_sequence(input_tensor, (h_0, c_0), [W_i, W_f, W_c, W_o, R_i, R_f, R_c, R_o, b_i, b_f, b_c, b_o])

        result0 = h_states.numpy()
        result1 = tf.stack([h_0, c_0]).numpy()
        
    return {"result0": result0, "result1": result1}

def main():
    A_TOL = 0.01

    input_size = 10
    hidden_size = 20
    batch_size = 5
    seq_len = 7
    num_layers = 1
    
    input_data = {
        "input": np.random.randn(seq_len, batch_size, input_size).astype(np.float32),
        "weight": [np.random.randn(hidden_size, input_size).astype(np.float32) for _ in range(12)],
        "hx": np.random.randn(2, batch_size, hidden_size).astype(np.float32),
        "seq_lengths": np.full((batch_size,), seq_len, dtype=np.int64),
        "batch_first": False,
        "dropout": 0.0,
        "train": True,
        "bidirectional": False,
        "batch_sizes": np.full((seq_len,), batch_size, dtype=np.int64),
        "mode": 0
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result0"], tf_result["result0"], atol=A_TOL), "Results0 do not match"
    assert np.allclose(torch_result["result1"], tf_result["result1"], atol=A_TOL), "Results1 do not match"

    print("Success")

if __name__ == "__main__":
    main()