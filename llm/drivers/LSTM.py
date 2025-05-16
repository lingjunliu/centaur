import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict["bias_ih"]) if "bias_ih" in input_dict else None
    bias_hh = torch.tensor(input_dict["bias_hh"]) if "bias_hh" in input_dict else None

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        if bias_ih is not None:
            bias_ih = bias_ih.cuda()
        if bias_hh is not None:
            bias_hh = bias_hh.cuda()

    lstm_cell = torch.nn.LSTM(input_size=input_tensor.shape[-1], hidden_size=weight_hh.shape[1]//4)
    lstm_cell.weight_ih_l0 = torch.nn.Parameter(weight_ih)
    lstm_cell.weight_hh_l0 = torch.nn.Parameter(weight_hh)
    if bias_ih is not None:
        lstm_cell.bias_ih_l0 = torch.nn.Parameter(bias_ih)
    if bias_hh is not None:
        lstm_cell.bias_hh_l0 = torch.nn.Parameter(bias_hh)
    
    hx = (torch.tensor(input_dict["hx"][0]).unsqueeze(0), torch.tensor(input_dict["hx"][1]).unsqueeze(0))
    if not cpu:
        hx = (hx[0].cuda(), hx[1].cuda())

    result = lstm_cell(input_tensor.unsqueeze(0), hx)
    
    if not cpu:
        result = (result[0].cpu(), result[1].cpu())

    return {"result": (result[0].detach().numpy(), (result[1][0].detach().numpy(), result[1][1].detach().numpy()))}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight_ih = tf.constant(input_dict["weight_ih"], dtype=tf.float32)
        weight_hh = tf.constant(input_dict["weight_hh"], dtype=tf.float32)
        bias_ih = tf.constant(input_dict["bias_ih"], dtype=tf.float32) if "bias_ih" in input_dict else None
        bias_hh = tf.constant(input_dict["bias_hh"], dtype=tf.float32) if "bias_hh" in input_dict else None

        hidden_size = weight_hh.shape[1] // 4
        
        
        lstm_cell = tf.keras.layers.LSTM(hidden_size, return_sequences=False, return_state=True,
                                     kernel_initializer=tf.keras.initializers.Constant(weight_ih.numpy()),
                                     recurrent_initializer=tf.keras.initializers.Constant(weight_hh.numpy()),
                                     bias_initializer=tf.keras.initializers.Constant(np.concatenate([input_dict["bias_ih"], input_dict["bias_hh"]]).reshape(1, -1)[0]))

        hx = (tf.constant(input_dict["hx"][0], dtype=tf.float32), tf.constant(input_dict["hx"][1], dtype=tf.float32))

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        output, h, c = lstm_cell(input_tensor, initial_state=hx)
        result = output.numpy(), (h.numpy(), c.numpy())
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_size = 10
    hidden_size = 5

    input_data = {
        "input": np.random.randn(1, input_size).astype(np.float32),
        "weight_ih": np.random.randn(input_size, hidden_size * 4).astype(np.float32),
        "weight_hh": np.random.randn(hidden_size, hidden_size * 4).astype(np.float32),
        "bias_ih": np.random.randn(hidden_size * 4).astype(np.float32),
        "bias_hh": np.random.randn(hidden_size * 4).astype(np.float32),
        "hx": (np.random.randn(1, hidden_size).astype(np.float32), np.random.randn(1, hidden_size).astype(np.float32))
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"][0], tf_result["result"][0], atol=A_TOL), "Output does not match"
    assert np.allclose(torch_result["result"][1][0], tf_result["result"][1][0], atol=A_TOL), "Hidden state does not match"
    assert np.allclose(torch_result["result"][1][1], tf_result["result"][1][1], atol=A_TOL), "Cell state does not match"

    print("Success")

if __name__ == "__main__":
    main()