import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()

    loss = torch.nn.BCELoss(weight=weight, reduction=reduction)
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
        weight = input_dict.get("weight", None)
        if weight is not None:
            weight = tf.constant(weight, dtype=tf.float32)
        size_average = input_dict.get("size_average", None)
        reduce = input_dict.get("reduce", None)
        reduction = input_dict.get("reduction", 'mean')

        def binary_cross_entropy(y_true, y_pred, weight=None, reduction='mean'):
            y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-7, clip_value_max=1 - 1e-7)
            bce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
            if weight is not None:
                bce = weight * bce
            if reduction == 'mean':
                return -tf.reduce_mean(bce)
            elif reduction == 'sum':
                return -tf.reduce_sum(bce)
            else:
                return -bce

        result = binary_cross_entropy(target_tensor, input_tensor, weight=weight, reduction=reduction)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.9, 0.2, 0.6, 0.1], dtype=np.float32),
        "target": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "weight": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.9, 0.2, 0.6, 0.1], dtype=np.float32),
        "target": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "weight": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        "reduction": 'sum'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.9, 0.2, 0.6, 0.1], dtype=np.float32),
        "target": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "reduction": 'none'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()