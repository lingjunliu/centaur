import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])
    observation = torch.tensor(input_dict["observation"])
    fake_quant_enabled = input_dict.get("fake_quant_enabled", True)
    momentum = input_dict.get("momentum", 0.1)
    eps = input_dict.get("eps", 1e-05)
    quant_min = input_dict.get("quant_min", 0)
    quant_max = input_dict.get("quant_max", 255)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        observation = observation.cuda()

    with torch.no_grad():
      if fake_quant_enabled:
          obs_min = torch.min(observation)
          obs_max = torch.max(observation)
          scale = (quant_max - quant_min) / (obs_max - obs_min + eps)
          zero_point = quant_min - obs_min * scale
          quantized = torch.clamp(torch.round(input_tensor * scale + zero_point), quant_min, quant_max)
          dequantized = (quantized - zero_point) / scale
          input_tensor = dequantized

      new_running_mean = running_mean * (1 - momentum) + torch.mean(observation) * momentum
      new_running_var = running_var * (1 - momentum) + torch.var(observation, unbiased=False) * momentum

      result = input_tensor

    if not cpu:
        result = result.cpu()
        new_running_mean = new_running_mean.cpu()
        new_running_var = new_running_var.cpu()
    
    return {
        "result": result.numpy(),
        "result1": new_running_mean.numpy(),
        "result2": new_running_var.numpy(),
    }

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        running_mean = tf.constant(input_dict["running_mean"])
        running_var = tf.constant(input_dict["running_var"])
        observation = tf.constant(input_dict["observation"])
        fake_quant_enabled = input_dict.get("fake_quant_enabled", True)
        momentum = input_dict.get("momentum", 0.1)
        eps = input_dict.get("eps", 1e-05)
        quant_min = input_dict.get("quant_min", 0)
        quant_max = input_dict.get("quant_max", 255)

        if fake_quant_enabled:
            obs_min = tf.math.reduce_min(observation)
            obs_max = tf.math.reduce_max(observation)
            scale = (quant_max - quant_min) / (obs_max - obs_min + eps)
            zero_point = quant_min - obs_min * scale
            quantized = tf.clip_by_value(tf.round(input_tensor * scale + zero_point), quant_min, quant_max)
            dequantized = (quantized - zero_point) / scale
            input_tensor = dequantized

        new_running_mean = running_mean * (1 - momentum) + tf.reduce_mean(observation) * momentum
        new_running_var = running_var * (1 - momentum) + tf.math.reduce_variance(observation) * momentum

        result = input_tensor.numpy()
        new_running_mean_np = new_running_mean.numpy()
        new_running_var_np = new_running_var.numpy()

    return {
        "result": result,
        "result1": new_running_mean_np,
        "result2": new_running_var_np,
    }

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "running_mean": np.array([0.0], dtype=np.float32),
        "running_var": np.array([1.0], dtype=np.float32),
        "observation": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "fake_quant_enabled": True,
        "momentum": 0.1,
        "eps": 1e-05,
        "quant_min": 0,
        "quant_max": 255,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["result1"], tf_result["result1"], atol=A_TOL), "Results1 do not match"
    assert np.allclose(torch_result["result2"], tf_result["result2"], atol=A_TOL), "Results2 do not match"

    print("Success")

if __name__ == "__main__":
    main()