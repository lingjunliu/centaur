import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    num_features = input_dict["num_features"]
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        
    bn = nn.BatchNorm1d(num_features, eps, momentum)
    relu = nn.ReLU()
    
    if 'weight' in input_dict:
        bn.weight = torch.nn.Parameter(torch.tensor(input_dict['weight']))
    if 'bias' in input_dict:
        bn.bias = torch.nn.Parameter(torch.tensor(input_dict['bias']))
    if 'running_mean' in input_dict:
        bn.running_mean = torch.tensor(input_dict['running_mean'])
    if 'running_var' in input_dict:
        bn.running_var = torch.tensor(input_dict['running_var'])
    if 'qconfig' in input_dict:
        pass 

    input_tensor = input_tensor.reshape(1, num_features, int(input_tensor.shape[0] / num_features))
    
    bn.eval()
    relu.eval()
    with torch.no_grad():
        bn_output = bn(input_tensor)
        result = relu(bn_output)
    
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        num_features = input_dict["num_features"]
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)

        if 'weight' in input_dict:
            weight = tf.constant(input_dict['weight'], dtype=tf.float32)
        else:
            weight = tf.ones((num_features,), dtype=tf.float32)
        if 'bias' in input_dict:
            bias = tf.constant(input_dict['bias'], dtype=tf.float32)
        else:
            bias = tf.zeros((num_features,), dtype=tf.float32)
        if 'running_mean' in input_dict:
            running_mean = tf.constant(input_dict['running_mean'], dtype=tf.float32)
        else:
            running_mean = tf.zeros((num_features,), dtype=tf.float32)
        if 'running_var' in input_dict:
            running_var = tf.constant(input_dict['running_var'], dtype=tf.float32)
        else:
            running_var = tf.ones((num_features,), dtype=tf.float32)
            
        input_tensor = tf.reshape(input_tensor, (1, num_features, int(input_tensor.shape[0] / num_features)))
        
        mean, variance = tf.nn.moments(input_tensor, axes=[0, 2], keepdims=True)

        scale = weight / tf.sqrt(running_var + eps)
        offset = bias - scale * running_mean
        
        normalized_input = (input_tensor - running_mean) / tf.sqrt(running_var + eps)
        output = scale * normalized_input + offset
        
        result = tf.nn.relu(output)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "num_features": 1,
        "weight": np.array([1.0], dtype=np.float32),
        "bias": np.array([0.0], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()