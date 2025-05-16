import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    h_0 = torch.tensor(input_dict["h_0"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict["bias_ih"])
    bias_hh = torch.tensor(input_dict["bias_hh"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        h_0 = h_0.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        bias_ih = bias_ih.cuda()
        bias_hh = bias_hh.cuda()

    params = [weight_ih, weight_hh, bias_ih, bias_hh]
    params_tuple = tuple(params)

    has_biases = True
    num_layers = 1
    dropout = 0.0
    train = False
    bidirectional = False
    batch_first = True

    result, _ = torch.gru(input_tensor, h_0, params=params_tuple, has_biases=has_biases, num_layers=num_layers, dropout=dropout, train=train, bidirectional=bidirectional, batch_first=batch_first)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        h_0 = tf.constant(input_dict["h_0"])
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias_ih = tf.constant(input_dict["bias_ih"])
        bias_hh = tf.constant(input_dict["bias_hh"])

        gru_cell = tf.keras.layers.GRUCell(weight_hh.shape[0])
        gru_cell.kernel = weight_ih
        gru_cell.recurrent_kernel = weight_hh
        gru_cell.bias = bias_ih + bias_hh

        output, state = tf.nn.dynamic_rnn(
            gru_cell,
            input_tensor,
            initial_state=h_0[0],
            dtype=tf.float32,
            time_major=False,
        )

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
        "h_0": np.random.rand(1, 2, 5).astype(np.float32),
        "weight_ih": np.random.rand(4, 5).astype(np.float32),
        "weight_hh": np.random.rand(5, 5).astype(np.float32),
        "bias_ih": np.random.rand(5).astype(np.float32),
        "bias_hh": np.random.rand(5).astype(np.float32)
    }
    
    # Modify h_0 to match expected input dimensions
    input_data["h_0"] = np.random.rand(1, 2, 5).astype(np.float32)
    
    # Ensure input_tensor is batch_first=True
    input_data["input"] = np.random.rand(2, 3, 4).astype(np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()