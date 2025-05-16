import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    delta = input_dict.get("delta", 1.0)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

    loss = torch.nn.HuberLoss(delta=delta, reduction=reduction)
    result = loss(input_tensor, target_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    target_tensor = tf.constant(input_dict["target"])
    delta = input_dict.get("delta", 1.0)
    reduction = input_dict.get("reduction", 'mean')

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        abs_diff = tf.abs(input_tensor - target_tensor)

        quadratic = tf.minimum(abs_diff, delta)
        linear = abs_diff - quadratic

        loss = 0.5 * quadratic**2 + delta * linear

        if reduction == 'sum':
            result = tf.reduce_sum(loss)
        elif reduction == 'mean':
            result = tf.reduce_mean(loss)
        else:
            result = tf.identity(loss)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "target": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "delta": 1.0,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "target": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "delta": 1.0,
        "reduction": 'sum'
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "target": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "delta": 2.0,
        "reduction": 'none'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()