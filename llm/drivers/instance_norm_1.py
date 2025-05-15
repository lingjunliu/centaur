import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    running_mean = torch.tensor(input_dict.get("running_mean", np.array([0.0], dtype=np.float32)))
    running_var = torch.tensor(input_dict.get("running_var", np.array([1.0], dtype=np.float32)))
    weight = torch.tensor(input_dict.get("weight", np.array([1.0], dtype=np.float32)))
    bias = torch.tensor(input_dict.get("bias", np.array([0.0], dtype=np.float32)))
    use_input_stats = input_dict.get("use_input_stats", True)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.instance_norm(
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        use_input_stats,
        momentum,
        eps,
        cudnn_enabled=True,
    )

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    input_tensor = tf.reshape(input_tensor, [1, 1, -1])
    running_mean = tf.constant(input_dict.get("running_mean", np.array([0.0], dtype=np.float32)), dtype=tf.float32)
    running_var = tf.constant(input_dict.get("running_var", np.array([1.0], dtype=np.float32)), dtype=tf.float32)
    weight = tf.constant(input_dict.get("weight", np.array([1.0])), dtype=tf.float32)
    bias = tf.constant(input_dict.get("bias", np.array([0.0])), dtype=tf.float32)
    use_input_stats = input_dict.get("use_input_stats", True)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if use_input_stats:
            mean, variance = tf.nn.moments(input_tensor, axes=[2], keepdims=True)
            mean = tf.reshape(mean, [])
            variance = tf.reshape(variance, [])
            
            if momentum is not None:
                running_mean_new = momentum * mean + (1 - momentum) * running_mean
                running_var_new = momentum * variance + (1 - momentum) * running_var
                running_mean = tf.Variable(running_mean_new)
                running_var = tf.Variable(running_var_new)

                
            else:
                running_mean_new = mean
                running_var_new = variance
                running_mean = tf.Variable(running_mean_new)
                running_var = tf.Variable(running_var_new)
        else:
            mean = running_mean
            variance = running_var
            running_mean_new = running_mean
            running_var_new = running_var

        normalized = (input_tensor - mean) / tf.sqrt(variance + eps)
        result = weight * normalized + bias
        result = tf.reshape(result, [-1]).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()