import numpy as np

def torch_version(input, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input["input"])
    hx = torch.tensor(input["hx"])
    cx = torch.tensor(input["cx"])
    
    lstm_cell = torch.nn.LSTMCell(input_size=input['input_size'], hidden_size=input['hidden_size'], bias=input['bias'])
    
    lstm_cell.weight_ih.data = torch.tensor(input['weight_ih'])
    lstm_cell.weight_hh.data = torch.tensor(input['weight_hh'])
    lstm_cell.bias_ih.data = torch.tensor(input['bias_ih'])
    lstm_cell.bias_hh.data = torch.tensor(input['bias_hh'])

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()
        cx = cx.cuda()
        lstm_cell = lstm_cell.cuda()
        lstm_cell.weight_ih.data = lstm_cell.weight_ih.data.cuda()
        lstm_cell.weight_hh.data = lstm_cell.weight_hh.data.cuda()
        lstm_cell.bias_ih.data = lstm_cell.bias_ih.data.cuda()
        lstm_cell.bias_hh.data = lstm_cell.bias_hh.data.cuda()

    hx, cx = lstm_cell(input_tensor, (hx, cx))
    
    return {"hx": hx.cpu().detach().numpy(), "cx": cx.cpu().detach().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    device_string = "/cpu:0" if cpu else "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input["input"])
        hx = tf.convert_to_tensor(input["hx"])
        cx = tf.convert_to_tensor(input["cx"])
        
        lstm_cell = tf.keras.layers.LSTMCell(units=input['hidden_size'], use_bias=input['bias'])
        
        # Create model and build weights
        lstm_cell.build(input_shape=(None, input['input_size']))
        weight_ih = np.array(input['weight_ih'])
        weight_hh = np.array(input['weight_hh'])
        
        # The weight shapes need to be transposed to match TensorFlow's LSTMCell
        lstm_cell.set_weights([
            tf.convert_to_tensor(weight_ih.T),
            tf.convert_to_tensor(weight_hh.T),
            tf.convert_to_tensor(input['bias_ih']) + tf.convert_to_tensor(input['bias_hh'])
        ])

        outputs, [hx, cx] = lstm_cell(input_tensor, [hx, cx])
        
        return {"hx": hx.numpy(), "cx": cx.numpy()}

def main():
    input_data = {
        "input": np.random.randn(1, 10).astype(np.float32),  # (batch, input_size)
        "hx": np.random.randn(1, 20).astype(np.float32),     # (batch, hidden_size)
        "cx": np.random.randn(1, 20).astype(np.float32),     # (batch, hidden_size)
        "input_size": 10,
        "hidden_size": 20,
        "bias": True,
        "weight_ih": np.random.randn(4 * 20, 10).astype(np.float32),  # (4*hidden_size, input_size)
        "weight_hh": np.random.randn(4 * 20, 20).astype(np.float32),  # (4*hidden_size, hidden_size)
        "bias_ih": np.random.randn(4 * 20).astype(np.float32),        # (4*hidden_size)
        "bias_hh": np.random.randn(4 * 20).astype(np.float32)         # (4*hidden_size)
    }
    
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    np.testing.assert_allclose(torch_result["hx"], tf_result["hx"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(torch_result["cx"], tf_result["cx"], rtol=1e-5, atol=1e-5)
    if np.allclose(torch_result["hx"], tf_result["hx"], rtol=1e-5, atol=1e-5) and np.allclose(torch_result["cx"], tf_result["cx"], rtol=1e-5, atol=1e-5):
        print("equal")
    else:
        print("not equal")
    
    # print("equal")

if __name__ == "__main__":
    main()