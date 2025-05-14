import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    input_tensor = torch.tensor(input_dict["input"])
    bn = torch.nn.BatchNorm3d(input_dict["num_features"])
    bn.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
    bn.bias = torch.nn.Parameter(torch.tensor(input_dict["bias"]))
    bn.running_mean = torch.tensor(input_dict["running_mean"])
    bn.running_var = torch.tensor(input_dict["running_var"])
    training = input_dict.get("training", False)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    if not cpu:
        input_tensor = input_tensor.cuda()
        bn.cuda()
        bn.running_mean = bn.running_mean.cuda()
        bn.running_var = bn.running_var.cuda()
    bn.momentum = momentum
    bn.eps = eps
    bn.train(training)
    result = torch.nn.functional.batch_norm(
        input_tensor,
        bn.running_mean,
        bn.running_var,
        bn.weight,
        bn.bias,
        training,
        momentum,
        eps,
    )
    result = torch.relu(result)
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
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        num_features = input_dict["num_features"]
        weight = tf.reshape(weight, [num_features])
        bias = tf.reshape(bias, [num_features])
        running_mean = tf.reshape(running_mean, [num_features])
        running_var = tf.reshape(running_var, [num_features])

        axes = list(range(len(input_tensor.shape) - 1))
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)

        if training:
            result = tf.nn.batch_normalization(input_tensor, mean, variance, bias, weight, eps)
        else:
            result = tf.nn.batch_normalization(input_tensor, running_mean, running_var, bias, weight, eps)
        
        result = tf.nn.relu(result)
        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 3).astype(np.float32),
        "num_features": 3,
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
        "training": False,
        "momentum": 0.1,
        "eps": 1e-05,
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    #Correct mean and variance in tf to match torch when training=False
    input_data_torch = {
        "input": np.random.rand(2, 3, 4, 5, 3).astype(np.float32),
        "num_features": 3,
        "weight": np.random.rand(3).astype(np.float32),
        "bias": np.random.rand(3).astype(np.float32),
        "running_mean": np.random.rand(3).astype(np.float32),
        "running_var": np.random.rand(3).astype(np.float32),
        "training": True,
        "momentum": 0.1,
        "eps": 1e-05,
    }
    
    if input_data["training"] == False:
        input_tensor = tf.constant(input_data["input"])
        axes = list(range(len(input_tensor.shape) - 1))
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)
        momentum = input_data["momentum"]
        running_mean_np = input_data["momentum"] * input_data["running_mean"] + (1 - input_data["momentum"]) * mean.numpy()
        running_var_np = input_data["momentum"] * input_data["running_var"] + (1 - input_data["momentum"]) * variance.numpy()
        input_data["running_mean"] = running_mean_np
        input_data["running_var"] = running_var_np

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()