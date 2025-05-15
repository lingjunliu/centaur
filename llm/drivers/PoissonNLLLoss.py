import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    log_input = input_dict.get("log_input", True)
    full = input_dict.get("full", False)
    eps = input_dict.get("eps", 1e-08)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

    loss = torch.nn.PoissonNLLLoss(log_input=log_input, full=full, eps=eps, reduction=reduction)
    result = loss(input_tensor, target_tensor)

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
        target_tensor = tf.constant(input_dict["target"], dtype=tf.float32)
        log_input = input_dict.get("log_input", True)
        full = input_dict.get("full", False)
        eps = input_dict.get("eps", 1e-08)
        reduction = input_dict.get("reduction", 'mean')

        if log_input:
            log_rate = input_tensor
        else:
            log_rate = tf.math.log(tf.maximum(input_tensor, eps))

        unreduced_loss = tf.math.exp(log_rate) - target_tensor * log_rate

        if full:
            unreduced_loss = unreduced_loss + tf.where(target_tensor > 0, target_tensor * tf.math.log(tf.clip_by_value(target_tensor, clip_value_min=eps, clip_value_max=tf.float32.max)) - target_tensor, tf.zeros_like(target_tensor, dtype=tf.float32))
        else:
            unreduced_loss = unreduced_loss

        if reduction == 'none':
            result = unreduced_loss
        elif reduction == 'mean':
            result = tf.reduce_mean(unreduced_loss)
        elif reduction == 'sum':
            result = tf.reduce_sum(unreduced_loss)
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "target": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "log_input": False,
        "full": False,
        "eps": 1e-08,
        "reduction": "mean"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.5, 1.0, 1.5], dtype=np.float32),
        "target": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "log_input": True,
        "full": False,
        "eps": 1e-08,
        "reduction": "mean"
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.5, 1.0, 1.5], dtype=np.float32),
        "target": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "log_input": True,
        "full": True,
        "eps": 1e-08,
        "reduction": "sum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.5, 1.0, 1.5], dtype=np.float32),
        "target": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "log_input": False,
        "full": True,
        "eps": 1e-08,
        "reduction": "none"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([0.5, 1.0, 1.5], dtype=np.float32),
        "target": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "log_input": False,
        "full": False,
        "eps": 1e-08,
        "reduction": "none"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()