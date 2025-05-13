import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight_tensor = torch.tensor(input_dict["weight"])
    bias_tensor = torch.tensor(input_dict["bias"]) if "bias" in input_dict else torch.tensor(np.zeros(input_dict["weight"].shape[0]))
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = weight_tensor.cuda()
        bias_tensor = bias_tensor.cuda()

    input_size = input_tensor.shape[-1]
    hidden_size = weight_tensor.shape[0]

    rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity='relu', batch_first=True)

    with torch.no_grad():
        rnn.weight_ih_l0 = torch.nn.Parameter(weight_tensor)
        rnn.weight_hh_l0 = torch.nn.Parameter(torch.zeros_like(rnn.weight_hh_l0))
        rnn.bias_ih_l0 = torch.nn.Parameter(bias_tensor)
        rnn.bias_hh_l0 = torch.nn.Parameter(torch.zeros_like(rnn.bias_hh_l0))
        result, _ = rnn(input_tensor.unsqueeze(0))
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result[0].detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight_tensor = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias_tensor = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else tf.zeros(input_dict["weight"].shape[0], dtype=tf.float32)

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        wx_b = tf.matmul(input_tensor, tf.transpose(weight_tensor)) + bias_tensor
        result = tf.nn.relu(wx_b)
        result = result.numpy()[0]

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "weight": np.array([[0.01, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8],
                           [0.9, 1.0, 1.1, 1.2]], dtype=np.float32),
        "bias": np.array([0.1, 0.2, 0.3], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()