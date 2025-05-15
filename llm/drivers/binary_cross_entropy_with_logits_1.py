import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    weight = torch.tensor(input_dict["weight"]) if "weight" in input_dict else None
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    reduction = input_dict.get("reduction", 'mean')
    pos_weight = torch.tensor(input_dict["pos_weight"]) if "pos_weight" in input_dict else None

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
        if weight is not None:
            weight = weight.cuda()
        if pos_weight is not None:
            pos_weight = pos_weight.cuda()

    result = torch.nn.functional.binary_cross_entropy_with_logits(
        input_tensor,
        target,
        weight=weight,
        reduction=reduction,
        pos_weight=pos_weight
    )

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
        target = tf.constant(input_dict["target"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32) if "weight" in input_dict else None
        reduction = input_dict.get("reduction", 'mean')
        pos_weight = tf.constant(input_dict["pos_weight"], dtype=tf.float32) if "pos_weight" in input_dict else None

        if pos_weight is not None:
            bce = tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=input_tensor, pos_weight=pos_weight)

            if weight is not None:
                result = bce * weight
            else:
                result = bce

            if reduction == 'mean':
                result = tf.reduce_mean(result)
            elif reduction == 'sum':
                result = tf.reduce_sum(result)
            else:
                pass
        else:
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=input_tensor)

            if weight is not None:
                result = bce * weight
            else:
                result = bce
            if reduction == 'mean':
                result = tf.reduce_mean(result)
            elif reduction == 'sum':
                result = tf.reduce_sum(result)
            else:
                pass

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.5, 0.8, 0.2], dtype=np.float32),
        "target": np.array([1.0, 0.0, 1.0], dtype=np.float32),
        "weight": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "pos_weight": np.array([2.0], dtype=np.float32),
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()