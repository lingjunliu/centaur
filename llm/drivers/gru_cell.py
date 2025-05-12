import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hidden_state = torch.tensor(input_dict["hidden_state"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict.get("bias_ih", np.zeros(input_dict["weight_ih"].shape[0] // 3).astype(np.float32)))
    bias_hh = torch.tensor(input_dict.get("bias_hh", np.zeros(input_dict["weight_hh"].shape[0] // 3).astype(np.float32)))
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_state = hidden_state.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        bias_ih = bias_ih.cuda()
        bias_hh = bias_hh.cuda()

    gru_cell = torch.nn.GRUCell(input_size=input_tensor.shape[-1], hidden_size=hidden_state.shape[-1])
    gru_cell.weight_ih = torch.nn.Parameter(weight_ih)
    gru_cell.weight_hh = torch.nn.Parameter(weight_hh)
    gru_cell.bias_ih = torch.nn.Parameter(bias_ih)
    gru_cell.bias_hh = torch.nn.Parameter(bias_hh)
    
    result = gru_cell(input_tensor, hidden_state)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    hidden_state = tf.constant(input_dict["hidden_state"])
    weight_ih = tf.constant(input_dict["weight_ih"])
    weight_hh = tf.constant(input_dict["weight_hh"])
    bias_ih = tf.constant(input_dict.get("bias_ih", np.zeros(input_dict["weight_ih"].shape[0] // 3).astype(np.float32)))
    bias_hh = tf.constant(input_dict.get("bias_hh", np.zeros(input_dict["weight_hh"].shape[0] // 3).astype(np.float32)))

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_size = input_tensor.shape[-1]
        hidden_size = hidden_state.shape[-1]

        w_ir, w_iz, w_in = tf.split(weight_ih, 3, axis=0)
        w_hr, w_hz, w_hn = tf.split(weight_hh, 3, axis=0)
        b_ir, b_iz, b_in = tf.split(bias_ih, 3, axis=0)
        b_hr, b_hz, b_hn = tf.split(bias_hh, 3, axis=0)

        r = tf.sigmoid(tf.matmul(input_tensor, tf.transpose(w_ir)) + tf.matmul(hidden_state, tf.transpose(w_hr)) + b_ir + b_hr)
        z = tf.sigmoid(tf.matmul(input_tensor, tf.transpose(w_iz)) + tf.matmul(hidden_state, tf.transpose(w_hz)) + b_iz + b_hz)
        n = tf.tanh(tf.matmul(input_tensor, tf.transpose(w_in)) + tf.matmul(r * hidden_state, tf.transpose(w_hn)) + b_in + b_hn)
        
        output = (1 - z) * n + z * hidden_state
        result = output.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_size = 2
    hidden_size = 3
    batch_size = 1

    input_data = {
        "input": np.random.rand(batch_size, input_size).astype(np.float32),
        "hidden_state": np.random.rand(batch_size, hidden_size).astype(np.float32),
        "weight_ih": np.random.rand(hidden_size * 3, input_size).astype(np.float32),
        "weight_hh": np.random.rand(hidden_size * 3, hidden_size).astype(np.float32),
    }
    input_data["bias_ih"] = np.random.rand(input_data["weight_ih"].shape[0]).astype(np.float32)
    input_data["bias_hh"] = np.random.rand(input_data["weight_hh"].shape[0]).astype(np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()