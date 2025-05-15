import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    anchor = torch.tensor(input_dict["anchor"])
    positive = torch.tensor(input_dict["positive"])
    negative = torch.tensor(input_dict["negative"])
    distance_function = input_dict.get("distance_function", lambda x, y: (x - y).norm(p=2, dim=1))
    margin = input_dict.get("margin", 1.0)
    swap = input_dict.get("swap", False)

    if not cpu:
        anchor = anchor.cuda()
        positive = positive.cuda()
        negative = negative.cuda()

    loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=distance_function,
        margin=margin,
        swap=swap
    )

    result = loss_fn(anchor, positive, negative)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    anchor = tf.constant(input_dict["anchor"])
    positive = tf.constant(input_dict["positive"])
    negative = tf.constant(input_dict["negative"])
    distance_function = input_dict.get("distance_function", lambda x, y: tf.norm(x - y, ord=2, axis=1))
    margin = input_dict.get("margin", 1.0)
    swap = input_dict.get("swap", False)

    def triplet_margin_with_distance_loss(anchor, positive, negative, distance_function, margin, swap):
        distance_positive = distance_function(anchor, positive)
        distance_negative = distance_function(anchor, negative)

        if swap:
            distance_positive_negative = distance_function(positive, negative)
            distance_positive = tf.minimum(distance_positive, distance_negative)
            distance_negative = tf.minimum(distance_positive_negative, distance_negative)
        
        losses = tf.maximum(distance_positive - distance_negative + margin, 0.0)
        return tf.reduce_mean(losses)

    if not cpu:
        with tf.device("/GPU:0"):
            result = triplet_margin_with_distance_loss(anchor, positive, negative, distance_function, margin, swap)
    else:
        with tf.device("/CPU:0"):
            result = triplet_margin_with_distance_loss(anchor, positive, negative, distance_function, margin, swap)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "anchor": np.array([[0.3, 0.7, 0.5], [0.3, 0.7, 0.5]], dtype=np.float32),
        "positive": np.array([[0.3, 0.7, 0.5], [0.3, 0.7, 0.5]], dtype=np.float32),
        "negative": np.array([[0.3, 0.7, 0.5], [0.3, 0.7, 0.5]], dtype=np.float32),
        "margin": 1.0,
        "swap": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()