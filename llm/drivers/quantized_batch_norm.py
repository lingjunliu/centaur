import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    use_input_stats = input_dict.get("use_input_stats", True)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    output_zero_point = int(input_dict.get("output_zero_point", 0))
    output_scale = float(input_dict.get("output_scale", 1.0))

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    q_input = torch.quantize_per_tensor(input_tensor, output_scale, output_zero_point, torch.quint8)

    with torch.no_grad():
        result = torch.nn.functional.batch_norm(q_input.float(), running_mean, running_var, weight, bias, use_input_stats, momentum, eps)

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
        use_input_stats = input_dict.get("use_input_stats", True)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)
        output_scale = input_dict.get("output_scale", 1.0)
        output_zero_point = input_dict.get("output_zero_point", 0)

        mean, variance = tf.nn.moments(input_tensor, axes=[]) if use_input_stats else (running_mean, running_var)

        normalized = (input_tensor - mean) / tf.sqrt(variance + eps)
        scaled = weight * normalized
        biased = scaled + bias

        result = biased

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32),
        "weight": np.array([1.0], dtype=np.float32),
        "bias": np.array([0.0], dtype=np.float32),
        "momentum": 0.1,
        "eps": 1e-05,
        "output_scale": 1.0,
        "output_zero_point": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()