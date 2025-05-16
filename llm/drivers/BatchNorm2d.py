import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    training = input_dict.get("training", False)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    num_features = input_tensor.shape[1]
    bn = torch.nn.BatchNorm2d(num_features)
    bn.running_mean = running_mean
    bn.running_var = running_var
    bn.weight = torch.nn.Parameter(weight)
    bn.bias = torch.nn.Parameter(bias)
    bn.momentum = momentum
    bn.eps = eps

    bn.train(training)
    result = bn(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        num_features = input_tensor.shape[1]

        if training:
            mean, variance = tf.nn.moments(input_tensor, axes=[0, 2, 3], keepdims=False)
            
            def update_mean_var(running_mean, running_var, mean, variance, momentum):
              new_running_mean = running_mean * (1 - momentum) + mean * momentum
              new_running_var = running_var * (1 - momentum) + variance * momentum
              return new_running_mean, new_running_var
            
            new_running_mean, new_running_var = update_mean_var(running_mean, running_var, mean, variance, momentum)


            def training_output(mean, variance, weight, bias, eps, input_tensor):
                normalized = (input_tensor - mean[None, :, None, None]) / tf.sqrt(variance[None, :, None, None] + eps)
                scaled = weight[None, :, None, None] * normalized + bias[None, :, None, None]
                return scaled, new_running_mean, new_running_var


            result, updated_running_mean, updated_running_var = training_output(mean, variance, weight, bias, eps, input_tensor)
            
        else:
            def inference_output(running_mean, running_var, weight, bias, eps, input_tensor):
                normalized = (input_tensor - running_mean[None, :, None, None]) / tf.sqrt(running_var[None, :, None, None] + eps)
                scaled = weight[None, :, None, None] * normalized + bias[None, :, None, None]
                return scaled

            result = inference_output(running_mean, running_var, weight, bias, eps, input_tensor)
            

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 4).astype(np.float32),
        "running_mean": np.zeros(3, dtype=np.float32),
        "running_var": np.ones(3, dtype=np.float32),
        "weight": np.ones(3, dtype=np.float32),
        "bias": np.zeros(3, dtype=np.float32),
        "training": False,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data["training"] = True
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()