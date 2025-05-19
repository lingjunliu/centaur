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

    if not cpu:
        anchor = anchor.cuda()
        positive = positive.cuda()
        negative = negative.cuda()

    loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=p, eps=eps, swap=swap)
    result = loss_fn(anchor, positive, negative)

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

        def compute_loss(anchor, positive, negative, margin, p, eps, swap):
            pos_dist = tf.norm(anchor - positive, ord=p, axis=-1)
            neg_dist = tf.norm(anchor - negative, ord=p, axis=-1)
            if swap:
                alt_pos_dist = tf.norm(anchor - negative, ord=p, axis=-1)
                alt_neg_dist = tf.norm(anchor - positive, ord=p, axis=-1)
                pos_dist = tf.minimum(pos_dist, alt_pos_dist)
                neg_dist = tf.minimum(neg_dist, alt_neg_dist)
            basic_loss = pos_dist - neg_dist + margin
            loss = tf.maximum(basic_loss, 0.0)
            return tf.reduce_mean(loss)

        result = compute_loss(anchor, positive, negative, margin, p, eps, swap).numpy()
        return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "anchor": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "positive": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "negative": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "margin": 1.0,
        "p": 2.0,
        "swap": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()