import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    anchor = torch.tensor(input_dict["anchor"])
    positive = torch.tensor(input_dict["positive"])
    negative = torch.tensor(input_dict["negative"])
    distance_function = input_dict.get("distance_function", lambda x, y: torch.sum(torch.abs(x - y), -1))
    margin = input_dict.get("margin", 1.0)
    swap = input_dict.get("swap", False)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        anchor = anchor.cuda()
        positive = positive.cuda()
        negative = negative.cuda()

    loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=margin, swap=swap, reduction=reduction)
    result = loss_fn(anchor, positive, negative)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    anchor = tf.constant(input_dict["anchor"])
    positive = tf.constant(input_dict["positive"])
    negative = tf.constant(input_dict["negative"])
    distance_function = input_dict.get("distance_function", lambda x, y: tf.reduce_sum(tf.abs(x - y), axis=-1))
    margin = input_dict.get("margin", 1.0)
    swap = input_dict.get("swap", False)
    reduction = input_dict.get("reduction", 'mean')

    def triplet_margin_with_distance_loss(anchor, positive, negative, distance_function, margin, swap, reduction):
        distance_positive = distance_function(anchor, positive)
        distance_negative = distance_function(anchor, negative)

        if swap:
            distance_swap = distance_function(positive, negative)
            distance_negative = tf.minimum(distance_negative, distance_swap)

        loss = tf.maximum(distance_positive - distance_negative + margin, 0.0)

        if reduction == 'mean':
            return tf.reduce_mean(loss)
        elif reduction == 'sum':
            return tf.reduce_sum(loss)
        else:
            return loss

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        result = triplet_margin_with_distance_loss(anchor, positive, negative, distance_function, margin, swap, reduction)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "anchor": np.array([[0.3, 0.7], [0.5, 0.5]], dtype=np.float32),
        "positive": np.array([[0.4, 0.6], [0.4, 0.6]], dtype=np.float32),
        "negative": np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32),
        "margin": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()