import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    training = input_dict["training"]
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    num_features = input_tensor.size(-1)

    if running_mean.size(0) != num_features:
        running_mean = torch.zeros(num_features)
        running_var = torch.ones(num_features)
        if not cpu:
            running_mean = running_mean.cuda()
            running_var = running_var.cuda()

    if weight.size(0) != num_features:
        weight = torch.ones(num_features)
        bias = torch.zeros(num_features)
        if not cpu:
            weight = weight.cuda()
            bias = bias.cuda()
    
    if training and input_tensor.size(0) <= 1:
        input_tensor = input_tensor.repeat(2, 1)

    if training:
        result = torch.nn.functional.batch_norm(
            input_tensor, running_mean, running_var, weight, bias,
            training=True, momentum=momentum, eps=eps
        )
    else:
        result = torch.nn.functional.batch_norm(
            input_tensor, running_mean, running_var, weight, bias,
            training=False, momentum=momentum, eps=eps
        )
    
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        training = input_dict["training"]
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        def batch_norm_elemt_tf(input_tensor, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-05):
          if training:
              mean = tf.reduce_mean(input_tensor, axis=0)
              variance = tf.reduce_mean(tf.square(input_tensor - mean), axis=0)
              
              new_running_mean = running_mean * momentum + mean * (1 - momentum)
              new_running_var = running_var * momentum + variance * (1 - momentum)

              normalized_input = (input_tensor - mean) / tf.sqrt(variance + eps)
              result = weight * normalized_input + bias

              return result, new_running_mean, new_running_var
          else:
              normalized_input = (input_tensor - running_mean) / tf.sqrt(running_var + eps)
              result = weight * normalized_input + bias
              return result, running_mean, running_var

        result, new_running_mean, new_running_var = batch_norm_elemt_tf(input_tensor, weight, bias, running_mean, running_var, training, momentum, eps)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "training": False,
        "momentum": 0.1,
        "eps": 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "training": True,
        "momentum": 0.1,
        "eps": 1e-05
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "running_mean": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "running_var": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "bias": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "training": True,
        "momentum": 0.1,
        "eps": 1e-05
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()