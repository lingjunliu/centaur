import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

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

    if training:
        def batch_norm_elementwise(input, weight, bias, running_mean, running_var, momentum, eps):
            exponential_average_factor = 0.0

            if training and running_mean is not None and running_var is not None:
                if running_mean.is_leaf and running_var.is_leaf:
                    exponential_average_factor = momentum
                else:
                    exponential_average_factor = 1.0 - momentum

            y = torch.batch_norm(
                input,
                running_mean,
                running_var,
                weight,
                bias,
                training,
                exponential_average_factor,
                eps,
                torch.backends.cudnn.enabled
            )
            return y

        result = batch_norm_elementwise(input_tensor, weight, bias, running_mean, running_var, momentum, eps)
    else:
        result = torch.batch_norm(
            input_tensor,
            running_mean,
            running_var,
            weight,
            bias,
            training,
            momentum,
            eps,
            torch.backends.cudnn.enabled
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

        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)

        def batch_norm_elemt(x, gamma, beta, mean, variance, training, momentum, epsilon):
            if training:
                m = tf.reduce_mean(x)
                v = tf.math.reduce_variance(x)

                new_mean = (1 - momentum) * mean + momentum * m
                new_variance = (1 - momentum) * variance + momentum * v
                
                x_normed = (x - m) / tf.sqrt(v + epsilon)
                output = gamma * x_normed + beta
                
                return output, new_mean, new_variance
            else:
                x_normed = (x - mean) / tf.sqrt(variance + epsilon)
                output = gamma * x_normed + beta
                return output
            
        running_mean_var = tf.Variable(running_mean, trainable=False)
        running_var_var = tf.Variable(running_var, trainable=False)
        
        def training_batch_norm(x, gamma, beta, mean, variance, momentum, epsilon):
          m = tf.reduce_mean(x)
          v = tf.math.reduce_variance(x)

          new_mean = (1 - momentum) * mean + momentum * m
          new_variance = (1 - momentum) * variance + momentum * v
          
          x_normed = (x - m) / tf.sqrt(v + epsilon)
          output = gamma * x_normed + beta
          return output, new_mean, new_variance
          
        def eval_batch_norm(x, gamma, beta, mean, variance, epsilon):
          x_normed = (x - mean) / tf.sqrt(variance + epsilon)
          output = gamma * x_normed + beta
          return output
          
        if training:
          result, new_mean, new_variance = training_batch_norm(input_tensor, weight, bias, running_mean_var, running_var_var, momentum, eps)
        else:
          result = eval_batch_norm(input_tensor, weight, bias, running_mean_var, running_var_var, eps)
            

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32),
        "weight": np.array([1.0], dtype=np.float32),
        "bias": np.array([0.0], dtype=np.float32),
        "training": False,
        "momentum": 0.1,
        "eps": 1e-05,
    }
    input_data["running_mean"] = np.zeros(1, dtype=np.float32)
    input_data["running_var"] = np.ones(1, dtype=np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32),
        "weight": np.array([1.0], dtype=np.float32),
        "bias": np.array([0.0], dtype=np.float32),
        "training": True,
        "momentum": 0.1,
        "eps": 1e-05,
    }
    input_data["running_mean"] = np.zeros(1, dtype=np.float32)
    input_data["running_var"] = np.ones(1, dtype=np.float32)
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()