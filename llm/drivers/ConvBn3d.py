import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    training = input_dict.get("training", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        if bias is not None:
            bias = bias.cuda()

    result_conv = torch.nn.functional.conv3d(input_tensor, weight, bias=None, stride=1, padding=(1,1,1), dilation=1, groups=1)
    
    C = weight.shape[0]

    result = torch.nn.functional.batch_norm(result_conv, running_mean, running_var, weight=torch.ones(C), bias=bias, training=training, momentum=momentum, eps=eps)


    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        training = input_dict.get("training", False)

        input_shape = input_tensor.shape
        weight_shape = weight.shape

        strides = [1, 1, 1, 1, 1]
        padding = 'VALID'
        
        #input_tensor_padded = tf.pad(input_tensor, [[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]], "CONSTANT")
        
        result_conv = tf.nn.conv3d(input_tensor, weight, strides=strides, padding='SAME')

        if training:
            mean, variance = tf.nn.moments(result_conv, axes=[0, 1, 2, 3])
            scale = tf.ones(input_dict["running_mean"].shape)
            if bias is not None:
                shift = tf.constant(input_dict["bias"])
            else:
                shift = tf.zeros(input_dict["running_mean"].shape[0], dtype=tf.float32)
                
            result = tf.nn.batch_normalization(result_conv, mean, variance, shift, scale, eps)
            
        else:
            scale = tf.ones(input_dict["running_mean"].shape)
            if bias is not None:
                shift = tf.constant(input_dict["bias"])
            else:
                shift = tf.zeros(input_dict["running_mean"].shape[0], dtype=tf.float32)
            result = tf.nn.batch_normalization(result_conv, running_mean, running_var, shift, scale, eps)
            
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "weight": np.random.rand(4, 3, 3, 3, 3).astype(np.float32),
        "bias": np.random.rand(4).astype(np.float32),
        "running_mean": np.random.rand(4).astype(np.float32),
        "running_var": np.random.rand(4).astype(np.float32),
        "eps": 1e-05,
        "momentum": 0.1,
        "training": False
    }
    
    input_data["input"] = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    input_data["weight"] = np.random.rand(4, 3, 3, 3, 3).astype(np.float32)
    input_data["bias"] = np.random.rand(4).astype(np.float32)
    input_data["running_mean"] = np.random.rand(4).astype(np.float32)
    input_data["running_var"] = np.random.rand(4).astype(np.float32)


    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()