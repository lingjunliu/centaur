import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    anchor = torch.tensor(input["anchor"], dtype=torch.float32, requires_grad=True)
    positive = torch.tensor(input["positive"], dtype=torch.float32, requires_grad=True)
    negative = torch.tensor(input["negative"], dtype=torch.float32, requires_grad=True)
    margin = input.get("margin", 1.0)
    p = input.get("p", 2.0)
    eps = input.get("eps", 1e-06)
    swap = input.get("swap", False)
    reduction = input.get("reduction", 'mean')

    if not cpu:
        anchor = anchor.cuda()
        positive = positive.cuda()
        negative = negative.cuda()
    
    # Apply to torch.nn.TripletMarginLoss
    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=p, eps=eps, swap=swap, reduction=reduction)
    loss = triplet_loss(anchor, positive, negative)

    if not cpu:
        loss = loss.cpu()

    return {"triplet_margin_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        anchor = tf.constant(input["anchor"])
        positive = tf.constant(input["positive"])
        negative = tf.constant(input["negative"])
        margin = input.get("margin", 1.0)
        p = input.get("p", 2.0)
        eps = input.get("eps", 1e-06)
        swap = input.get("swap", False)
        reduction = input.get("reduction", 'mean')

        # Implementing the equivalent using TensorFlow operations
        def pairwise_distance(x, y, p=2.0, eps=1e-06):
            return tf.norm(x - y, ord=p, axis=-1, keepdims=False) + eps

        distance_ap = pairwise_distance(anchor, positive, p, eps)
        distance_an = pairwise_distance(anchor, negative, p, eps)
        
        if swap:
            distance_pn = pairwise_distance(positive, negative, p, eps)
            pairwise_neg = tf.minimum(distance_an, distance_pn)
        else:
            pairwise_neg = distance_an

        loss = tf.maximum(distance_ap - pairwise_neg + margin, 0.0)

        if reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)

        return {"triplet_margin_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "anchor": np.random.randn(100, 128).astype(np.float32),
        "positive": np.random.randn(100, 128).astype(np.float32),
        "negative": np.random.randn(100, 128).astype(np.float32),
        "margin": 1.0,
        "p": 2.0,
        "eps": 1e-06,
        "swap": False,
        "reduction": 'mean'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check if results are equal
    np_torch_result = np.asarray(torch_result["triplet_margin_loss"], dtype=np.float32)
    np_tf_result = np.asarray(tf_result["triplet_margin_loss"], dtype=np.float32)
    
    if np.allclose(np_torch_result, np_tf_result, rtol=1e-05):
        print("equal")
    else:
        print("not equal")
    
if __name__ == "__main__":
    main()