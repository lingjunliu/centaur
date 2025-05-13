import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])
    weight_ih = torch.tensor(input_dict["weight_ih"])
    weight_hh = torch.tensor(input_dict["weight_hh"])
    bias_ih = torch.tensor(input_dict["bias_ih"]) if "bias_ih" in input_dict else None
    bias_hh = torch.tensor(input_dict["bias_hh"]) if "bias_hh" in input_dict else None

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()
        weight_ih = weight_ih.cuda()
        weight_hh = weight_hh.cuda()
        if bias_ih is not None:
            bias_ih = bias_ih.cuda()
        if bias_hh is not None:
            bias_hh = bias_hh.cuda()

    rnn_cell = torch.nn.RNNCell(input_size=input_tensor.shape[1], hidden_size=hx.shape[1])
    rnn_cell.weight_ih = torch.nn.Parameter(weight_ih)
    rnn_cell.weight_hh = torch.nn.Parameter(weight_hh)
    if bias_ih is not None:
        rnn_cell.bias_ih = torch.nn.Parameter(bias_ih)
    if bias_hh is not None:
        rnn_cell.bias_hh = torch.nn.Parameter(bias_hh)
    
    result = rnn_cell(input_tensor, hx)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hx = tf.constant(input_dict["hx"])
        weight_ih = tf.constant(input_dict["weight_ih"])
        weight_hh = tf.constant(input_dict["weight_hh"])
        bias_ih = tf.constant(input_dict["bias_ih"]) if "bias_ih" in input_dict else None
        bias_hh = tf.constant(input_dict["bias_hh"]) if "bias_hh" in input_dict else None

        Wh = weight_ih
        Uh = weight_hh
        bh = bias_ih if bias_ih is not None else tf.zeros([weight_ih.shape[0]])
        ch = bias_hh if bias_hh is not None else tf.zeros([weight_hh.shape[0]])
    
        pre_h = tf.matmul(input_tensor, tf.transpose(Wh)) + bh + tf.matmul(hx, tf.transpose(Uh)) + ch
        result = tf.tanh(pre_h)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_size = 5
    hidden_size = 10
    batch_size = 2
    
    input_data = {
        "input": np.random.rand(batch_size, input_size).astype(np.float32),
        "hx": np.random.rand(batch_size, hidden_size).astype(np.float32),
        "weight_ih": np.random.rand(hidden_size, input_size).astype(np.float32),
        "weight_hh": np.random.rand(hidden_size, hidden_size).astype(np.float32),
        "bias_ih": np.random.rand(hidden_size).astype(np.float32),
        "bias_hh": np.random.rand(hidden_size).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()