import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    weight = torch.tensor(input_dict["weight"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    bias = torch.tensor(input_dict["bias"], dtype=torch.float32) if "bias" in input_dict else None
    running_mean = torch.tensor(input_dict["running_mean"], dtype=torch.float32)
    running_var = torch.tensor(input_dict["running_var"], dtype=torch.float32)
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    training = input_dict.get("training", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda() if bias is not None else None
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
    
    conv = torch.nn.functional.conv2d(input_tensor, weight, bias=bias, stride=(1,1), padding=(0,0))
    
    if training:
        bn = torch.nn.functional.batch_norm(
            conv,
            running_mean,
            running_var,
            weight=None,
            bias=None,
            training=True,
            momentum=momentum,
            eps=eps
        )
    else:
        normalized_input = (conv - running_mean.view(1, -1, 1, 1)) / torch.sqrt(running_var.view(1, -1, 1, 1) + eps)
        if weight is not None:
            bn = normalized_input * weight.view(1, -1, 1, 1)
        else:
             bn = normalized_input
    

    if not cpu:
        bn = bn.cpu()
    
    return {"result": bn.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32) if "bias" in input_dict else None
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        training = input_dict.get("training", False)
        
        input_tensor_expanded = tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=0)
        weight_expanded = tf.expand_dims(tf.expand_dims(weight, axis=0), axis=0)
        
        conv = tf.nn.conv2d(input_tensor_expanded, weight_expanded, strides=[1, 1, 1, 1], padding='VALID')
        
        if bias is not None:
            conv = tf.nn.bias_add(conv, bias)

        if training:
            mean, variance = tf.nn.moments(conv, axes=[0, 1, 2])
            
            def update_mean_var():
                new_running_mean = momentum * mean + (1 - momentum) * running_mean
                new_running_var = momentum * variance + (1 - momentum) * running_var
                return new_running_mean, new_running_var
            
            running_mean, running_var = update_mean_var()

            scale = tf.cast(tf.math.rsqrt(variance + eps), tf.float32)
            normalized_input = (conv - mean) * scale
        else:
            scale = tf.cast(tf.math.rsqrt(running_var + eps), tf.float32)
            normalized_input = (conv - running_mean) * scale

        result = tf.reshape(normalized_input, [-1]).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "weight": np.array([0.5], dtype=np.float32),
        "bias": np.array([0.1], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "training": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()