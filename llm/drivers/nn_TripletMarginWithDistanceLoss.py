import numpy as np
import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F

def torch_version(input_dict, cpu=True):
    anchor = torch.tensor(input_dict["anchor"])
    positive = torch.tensor(input_dict["positive"])
    negative = torch.tensor(input_dict["negative"])
    distance_function = input_dict.get("distance_function", None)
    margin = input_dict.get("margin", 1.0)
    swap = input_dict.get("swap", False)
    reduction = input_dict.get("reduction", 'mean')

    if distance_function is None:
        distance_function = nn.PairwiseDistance()

    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=margin, swap=swap, reduction=reduction)
    
    if not cpu:
        anchor = anchor.cuda()
        positive = positive.cuda()
        negative = negative.cuda()

    output = triplet_loss(anchor, positive, negative)

    if not cpu:
        output = output.cpu()

    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    anchor = tf.constant(input_dict["anchor"])
    positive = tf.constant(input_dict["positive"])
    negative = tf.constant(input_dict["negative"])
    distance_function = input_dict.get("distance_function", None)
    margin = input_dict.get("margin", 1.0)
    swap = input_dict.get("swap", False)
    reduction = input_dict.get("reduction", 'mean')

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if distance_function is None:
            distance_function = lambda x, y: tf.norm(x - y, axis=-1)

        d_ap = distance_function(anchor, positive)
        d_an = distance_function(anchor, negative)

        if swap:
            d_pn = distance_function(positive, negative)
            swap_mask = d_pn < d_ap
            d_ap_swapped = tf.where(swap_mask, d_pn, d_ap)
            d_an_swapped = tf.where(swap_mask, d_pn, d_an)
            loss = tf.maximum(d_ap_swapped - d_an_swapped + margin, 0.0)
        else:
            loss = tf.maximum(d_ap - d_an + margin, 0.0)

        if reduction == 'none':
            pass
        elif reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
        
        if reduction == 'none':
            loss = loss.numpy()
        elif reduction == 'mean':
            loss = loss.numpy()
        else:
            loss = loss.numpy()
    return {"result": loss}

def main():
    A_TOL = 0.01
    input_data = {
        "anchor": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "positive": np.array([[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]], dtype=np.float32),
        "negative": np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float32),
        "margin": 0.5,
        "swap": False,
        "reduction": 'mean',
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "anchor": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "positive": np.array([[0.15, 0.25, 0.35]], dtype=np.float32),
        "negative": np.array([[0.2, 0.3, 0.4]], dtype=np.float32),
        "margin": 0.5,
        "swap": True,
        "reduction": 'none',
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()