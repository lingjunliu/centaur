import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    var = torch.tensor(input_dict["var"])
    full = input_dict.get("full", False)
    eps = input_dict.get("eps", 1e-06)
    reduction = input_dict.get("reduction", "mean")
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
        var = var.cuda()
    
    loss = torch.nn.GaussianNLLLoss(full=full, eps=eps, reduction=reduction)
    result = loss(input_tensor, target, var)
    
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
        target = tf.constant(input_dict["target"], dtype=tf.float32)
        var = tf.constant(input_dict["var"], dtype=tf.float32)
        full = input_dict.get("full", False)
        eps = input_dict.get("eps", 1e-06)
        reduction = input_dict.get("reduction", "mean")
        
        elementwise_loss = 0.5 * (tf.math.log(var) + tf.math.divide(tf.math.squared_difference(input_tensor, target), var))
        
        if full:
            elementwise_loss = elementwise_loss + 0.5 * np.log(2 * np.pi)
        
        if reduction == "none":
            result = elementwise_loss
        elif reduction == "mean":
            result = tf.reduce_mean(elementwise_loss)
        elif reduction == "sum":
            result = tf.reduce_sum(elementwise_loss)
        else:
            raise ValueError("Invalid reduction option")

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "target": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "var": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "full": False,
        "eps": 1e-06,
        "reduction": "mean"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()