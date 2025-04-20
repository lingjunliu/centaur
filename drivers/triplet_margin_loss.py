import numpy as np

def torch_version(input, cpu=True):
    import torch
    
    # Unpack input dictionary
    anchor = torch.tensor(input["anchor"])
    positive = torch.tensor(input["positive"])
    negative = torch.tensor(input["negative"])
    margin = input.get("margin", 1.0)
    p = input.get("p", 2)
    eps = input.get("eps", 1e-06)
    swap = input.get("swap", False)
    reduction = input.get("reduction", 'mean')

    # Apply to torch.nn.functional.triplet_margin_loss
    loss = torch.nn.functional.triplet_margin_loss(
        anchor, positive, negative, margin=margin, p=p, eps=eps,
        swap=swap, reduction=reduction
    )

    if not cpu:
        loss = loss.cpu()

    return {"triplet_margin_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

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
        p = input.get("p", 2)
        eps = input.get("eps", 1e-06)
        swap = input.get("swap", False)
        reduction = input.get("reduction", 'mean')

        def _pairwise_distances(a, b):
            """ Compute pairwise distances between `a` and `b`."""
            diff = tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)
            if p == 2:
                distances = tf.reduce_sum(tf.square(diff), axis=2)
            else:
                distances = tf.reduce_sum(tf.abs(diff) ** p, axis=2)
            return distances

        # Calculate distances
        pos_dist = tf.norm(anchor - positive, ord=p, axis=1)
        neg_dist = tf.norm(anchor - negative, ord=p, axis=1)
        if swap:
            neg_dist_swap = tf.norm(positive - negative, ord=p, axis=1)
            neg_dist = tf.minimum(neg_dist, neg_dist_swap)

        # Compute triplet loss
        loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)

        if reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)

        return {"triplet_margin_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "anchor": np.array([[0.5, 0.5], [0.2, 0.6]], dtype=np.float32),
        "positive": np.array([[0.45, 0.55], [0.25, 0.65]], dtype=np.float32),
        "negative": np.array([[0.8, 0.9], [0.8, 0.95]], dtype=np.float32),
        "margin": 1.0,
        "p": 2,
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

    # Check equality of results
    if np.isclose(torch_result["triplet_margin_loss"], tf_result["triplet_margin_loss"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()