import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input = torch.tensor(input_dict["input"])
    h_in = torch.tensor(input_dict["h_in"])
    w_ih = torch.tensor(input_dict["w_ih"])
    w_hh = torch.tensor(input_dict["w_hh"])
    b_ih = torch.tensor(input_dict.get("b_ih", np.zeros(w_ih.shape[0])), dtype=torch.float32)
    b_hh = torch.tensor(input_dict.get("b_hh", np.zeros(w_hh.shape[0])), dtype=torch.float32)
    
    if not cpu:
        input = input.cuda()
        h_in = h_in.cuda()
        w_ih = w_ih.cuda()
        w_hh = w_hh.cuda()
        b_ih = b_ih.cuda()
        b_hh = b_hh.cuda()

    hidden_size = h_in.shape[1]
    input_size = input.shape[1]
    gru_cell = torch.nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
    gru_cell.weight_ih = torch.nn.Parameter(w_ih.T)
    gru_cell.weight_hh = torch.nn.Parameter(w_hh.T)
    gru_cell.bias_ih = torch.nn.Parameter(b_ih)
    gru_cell.bias_hh = torch.nn.Parameter(b_hh)

    result = gru_cell(input, h_in)

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
        h_in = tf.constant(input_dict["h_in"])
        w_ih = tf.constant(input_dict["w_ih"])
        w_hh = tf.constant(input_dict["w_hh"])
        b_ih = tf.constant(input_dict.get("b_ih", np.zeros(w_ih.shape[0])), dtype=w_ih.dtype)
        b_hh = tf.constant(input_dict.get("b_hh", np.zeros(w_hh.shape[0])), dtype=w_hh.dtype)

        gru_cell = tf.keras.layers.GRUCell(units=h_in.shape[1])
        gru_cell.kernel = w_ih.numpy()
        gru_cell.recurrent_kernel = w_hh.numpy()
        gru_cell.bias = b_ih.numpy() + b_hh.numpy()

        result = gru_cell(input_tensor, states=[h_in])[0].numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2]], dtype=np.float32),
        "h_in": np.array([[0.3, 0.4]], dtype=np.float32),
        "w_ih": np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]], dtype=np.float32),
        "w_hh": np.array([[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32),
        "b_ih": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        "b_hh": np.array([0.4, 0.5, 0.6, 0.1, 0.2, 0.3], dtype=np.float32)
    }
    
    hidden_size = input_data["h_in"].shape[1]
    input_size = input_data["input"].shape[1]
    
    w_ih = input_data["w_ih"]
    w_ih = w_ih.reshape(3*hidden_size, input_size, order='F')
    input_data["w_ih"] = w_ih
    
    w_hh = input_data["w_hh"]
    w_hh = w_hh.reshape(3*hidden_size, hidden_size, order='F')
    input_data["w_hh"] = w_hh

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()