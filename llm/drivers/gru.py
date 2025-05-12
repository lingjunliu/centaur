import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hidden_state = torch.tensor(input_dict["hx"])
    input_size = input_dict["input_size"]
    hidden_size = input_dict["hidden_size"]
    num_layers = input_dict.get("num_layers", 1)
    batch_first = input_dict.get("batch_first", False)
    dropout = input_dict.get("dropout", 0.0)
    bidirectional = input_dict.get("bidirectional", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_state = hidden_state.cuda()

    gru = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                       batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    result, hn = gru(input_tensor, hidden_state)
    
    if not cpu:
        result = result.cpu()
        hn = hn.cpu()
        
    return {"result": result.detach().numpy(), "hn": hn.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        hidden_state = tf.constant(input_dict["hx"], dtype=tf.float32)
        input_size = input_dict["input_size"]
        hidden_size = input_dict["hidden_size"]
        num_layers = input_dict.get("num_layers", 1)
        batch_first = input_dict.get("batch_first", False)
        dropout = input_dict.get("dropout", 0.0)
        bidirectional = input_dict.get("bidirectional", False)

        if batch_first:
            input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2])

        gru_cells = [tf.keras.layers.GRUCell(hidden_size) for _ in range(num_layers)]
        stacked_gru = tf.keras.layers.StackedRNNCells(gru_cells)
        gru = tf.keras.layers.RNN(stacked_gru, return_sequences=True, return_state=True)

        # Reshape hidden_state for TensorFlow GRU
        initial_state = tf.split(hidden_state, num_or_size_splits=num_layers, axis=0)
        initial_state = [tf.squeeze(s, axis=[0,1]) for s in initial_state] #Squeeze along both 0 and 1


        outputs, last_state = gru(input_tensor, initial_state=initial_state)

        if batch_first:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
        
        result = outputs.numpy()
        hn = np.stack(last_state, axis=0).numpy()

    return {"result": result, "hn": hn}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5, 3, 10).astype(np.float32),
        "hx": np.random.rand(1, 3, 20).astype(np.float32),
        "input_size": 10,
        "hidden_size": 20,
        "num_layers": 1,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["hn"], tf_result["hn"], atol=A_TOL), "Hidden states do not match"

    print("Success")

if __name__ == "__main__":
    main()