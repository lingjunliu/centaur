import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])
    batch_sizes = torch.tensor(input_dict["batch_sizes"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()
        batch_sizes = batch_sizes.cuda()
    
    
    packed_input = pack_padded_sequence(input_tensor, batch_sizes.cpu(), enforce_sorted=False)
    
    
    rnn = torch.nn.RNN(input_size=input_tensor.shape[2], hidden_size=hx.shape[2], nonlinearity='tanh', batch_first=False)
    
    

    packed_output, hn = rnn(packed_input, hx)
    
    output, _ = pad_packed_sequence(packed_output, batch_first=False)
    

    if not cpu:
        output = output.cpu()
    
    return {"result": output.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hx = tf.constant(input_dict["hx"])
        batch_sizes = tf.constant(input_dict["batch_sizes"])

        num_batches = batch_sizes.numpy().tolist()
        
        outputs = []
        start_index = 0

        hx_state = hx[0] 

        for i, batch_size in enumerate(num_batches):
            end_index = start_index + batch_size
            input_slice = input_tensor[start_index:end_index]
            
            
            output = tf.tanh(tf.matmul(input_slice, tf.eye(input_tensor.shape[2], dtype=input_tensor.dtype)) + hx_state)
            outputs.append(output)
            
            hx_state = output[-1] 
            start_index = end_index

        result = tf.concat(outputs, axis=0)
        result = tf.reshape(result, [input_tensor.shape[0], 1, input_tensor.shape[2]])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]], dtype=np.float32),
        "hx": np.array([[[0.7, 0.8]]], dtype=np.float32),
        "batch_sizes": np.array([3], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    print("Torch:", torch_result["result"])
    print("TF:", tf_result["result"])

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()