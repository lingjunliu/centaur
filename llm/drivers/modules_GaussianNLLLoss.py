import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    var_tensor = torch.tensor(input_dict["var"])
    full = input_dict.get("full", False)
    eps = input_dict.get("eps", 1e-6)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        var_tensor = var_tensor.cuda()

    loss = torch.nn.GaussianNLLLoss(full=full, eps=eps, reduction=reduction)
    result = loss(input_tensor, target_tensor, var_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        target_tensor = tf.constant(input_dict["target"], dtype=tf.float32)
        var_tensor = tf.constant(input_dict["var"], dtype=tf.float32)
        full = input_dict.get("full", False)
        eps = input_dict.get("eps", 1e-6)
        reduction = input_dict.get("reduction", 'mean')

        logvar = tf.math.log(var_tensor)
        loss = 0.5 * (logvar + tf.math.divide(tf.math.squared_difference(input_tensor, target_tensor), var_tensor))

        if full:
            loss += 0.5 * tf.math.log(2 * np.pi)
        
        if reduction == 'mean':
            result = tf.reduce_mean(loss)
        elif reduction == 'sum':
            result = tf.reduce_sum(loss)
        else:
            result = loss

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "target": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "var": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "reduction": "mean"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "target": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "var": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "reduction": "sum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "target": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "var": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "reduction": "none"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()