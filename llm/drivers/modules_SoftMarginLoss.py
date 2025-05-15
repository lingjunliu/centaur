import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])

    reduction = input_dict.get("reduction", "mean")

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

    loss_fn = torch.nn.SoftMarginLoss(reduction=reduction)
    result = loss_fn(input_tensor, target_tensor)

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
        reduction = input_dict.get("reduction", "mean")

        result = tf.nn.softplus(tf.multiply(-target_tensor, input_tensor))

        if reduction == "mean":
            result = tf.reduce_mean(result)
        elif reduction == "sum":
            result = tf.reduce_sum(result)
        else:
            pass

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.5, -0.2, 0.7], dtype=np.float32),
        "target": np.array([1.0, -1.0, 1.0], dtype=np.float32),
        "reduction": "mean"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.5, -0.2, 0.7], dtype=np.float32),
        "target": np.array([1.0, -1.0, 1.0], dtype=np.float32),
        "reduction": "sum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.5, -0.2, 0.7], dtype=np.float32),
        "target": np.array([1.0, -1.0, 1.0], dtype=np.float32),
        "reduction": "none"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()