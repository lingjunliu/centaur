import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    quant_min = input_dict.get("quant_min", 0)
    quant_max = input_dict.get("quant_max", 255)
    qscheme = input_dict.get("qscheme", torch.per_tensor_affine)
    fake_quant_enabled = input_dict.get("fake_quant_enabled", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.fused_moving_avg_obs_fake_quant(
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        torch.tensor(momentum),
        torch.tensor(eps),
        quant_min,
        quant_max,
        qscheme,
        fake_quant_enabled,
    )

    if not cpu:
        result = (result[0].cpu(), result[1].cpu(), result[2].cpu())

    return {
        "result": result[0].numpy(),
        "running_mean": result[1].numpy(),
        "running_var": result[2].numpy(),
    }

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        running_mean = tf.Variable(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.Variable(input_dict["running_var"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)
        quant_min = input_dict.get("quant_min", 0)
        quant_max = input_dict.get("quant_max", 255)
        qscheme = input_dict.get("qscheme", torch.per_tensor_affine)
        fake_quant_enabled = input_dict.get("fake_quant_enabled", True)

        mean_update = running_mean.assign(momentum * running_mean + (1 - momentum) * tf.reduce_mean(input_tensor))
        var_update = running_var.assign(momentum * running_var + (1 - momentum) * tf.math.reduce_variance(input_tensor))

        def fake_quantize(x, quant_min, quant_max):
            min_val = tf.reduce_min(x)
            max_val = tf.reduce_max(x)
            scale = (quant_max - quant_min) / (max_val - min_val + eps)
            zero_point = quant_min - min_val * scale
            x_scaled = x * scale + zero_point
            x_clamped = tf.clip_by_value(x_scaled, quant_min, quant_max)
            return x_clamped

        if fake_quant_enabled:
            quantized_tensor = fake_quantize(input_tensor, quant_min, quant_max)
        else:
            quantized_tensor = input_tensor

        output_tensor = quantized_tensor * weight + bias

        result = output_tensor.numpy()
        running_mean_result = running_mean.numpy()
        running_var_result = running_var.numpy()

    return {
        "result": result,
        "running_mean": running_mean_result,
        "running_var": running_var_result,
    }

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "running_mean": np.array(0.0, dtype=np.float32),
        "running_var": np.array(1.0, dtype=np.float32),
        "weight": np.array(1.0, dtype=np.float32),
        "bias": np.array(0.0, dtype=np.float32),
        "momentum": 0.1,
        "eps": 1e-05,
        "quant_min": 0,
        "quant_max": 255,
        "qscheme": torch.per_tensor_affine,
        "fake_quant_enabled": True,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match for result"
    assert np.allclose(torch_result["running_mean"], tf_result["running_mean"], atol=A_TOL), "Results do not match for running_mean"
    assert np.allclose(torch_result["running_var"], tf_result["running_var"], atol=A_TOL), "Results do not match for running_var"


    print("Success")

if __name__ == "__main__":
    main()