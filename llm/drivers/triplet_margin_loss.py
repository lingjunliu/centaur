import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    anchor = torch.tensor(input_dict["anchor"])
    positive = torch.tensor(input_dict["positive"])
    negative = torch.tensor(input_dict["negative"])
    margin = input_dict.get("margin", 1.0)
    p = input_dict.get("p", 2.0)
    eps = input_dict.get("eps", 1e-6)
    swap = input_dict.get("swap", False)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        anchor = anchor.cuda()
        positive = positive.cuda()
        negative = negative.cuda()

    result = torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=margin, p=p, eps=eps, swap=swap, reduction=reduction)

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
        anchor = tf.constant(input_dict["anchor"])
        positive = tf.constant(input_dict["positive"])
        negative = tf.constant(input_dict["negative"])
        margin = input_dict.get("margin", 1.0)
        p = input_dict.get("p", 2.0)
        eps = input_dict.get("eps", 1e-6)
        swap = input_dict.get("swap", False)
        reduction = input_dict.get("reduction", 'mean')

        pos_dist = tf.norm(anchor - positive, ord=p, axis=None)
        neg_dist = tf.norm(anchor - negative, ord=p, axis=None)

        if swap:
            pos_dist_swap = tf.norm(anchor - negative, ord=p, axis=None)
            neg_dist_swap = tf.norm(anchor - positive, ord=p, axis=None)
            distance = tf.minimum(tf.maximum(pos_dist - neg_dist + margin, 0.0), tf.maximum(pos_dist_swap - neg_dist_swap + margin, 0.0))
        else:
            distance = tf.maximum(pos_dist - neg_dist + margin, 0.0)

        if reduction == 'mean':
            result = tf.reduce_mean(distance).numpy()
        elif reduction == 'sum':
            result = tf.reduce_sum(distance).numpy()
        else:
            result = distance.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "anchor": np.array([0.3, 0.7, 0.5], dtype=np.float32),
        "positive": np.array([0.4, 0.7, 0.5], dtype=np.float32),
        "negative": np.array([0.1, 0.3, 0.9], dtype=np.float32),
        "margin": 0.5,
        "p": 2,
        "swap": False,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()