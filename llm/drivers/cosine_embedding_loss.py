import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.functional as F

def torch_version(input_dict, cpu=True):

    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    target = torch.tensor(input_dict["target"])
    margin = input_dict.get("margin", 0.0)
    reduction = input_dict.get("reduction", "mean")

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()

    if reduction == "mean":
        reduction_mode = 1
    elif reduction == "sum":
        reduction_mode = 0
    else:
        reduction_mode = 2

    result = torch.cosine_embedding_loss(input1, input2, target, margin=margin, reduction=reduction_mode)

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
        input1 = tf.constant(input_dict["input1"], dtype=tf.float32)
        input2 = tf.constant(input_dict["input2"], dtype=tf.float32)
        target = tf.constant(input_dict["target"], dtype=tf.float32)
        margin = input_dict.get("margin", 0.0)
        reduction = input_dict.get("reduction", "mean")

        input1_norm = tf.linalg.norm(input1, axis=1, keepdims=True)
        input1_normalized = input1 / input1_norm
        input2_norm = tf.linalg.norm(input2, axis=1, keepdims=True)
        input2_normalized = input2 / input2_norm
        
        similarity = tf.reduce_sum(input1_normalized * input2_normalized, axis=1)
        
        loss = target * (1 - similarity) + (1 - target) * tf.maximum(0.0, similarity - margin)

        if reduction == "mean":
            result = tf.reduce_mean(loss)
        elif reduction == "sum":
            result = tf.reduce_sum(loss)
        else:
            result = loss

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input1": np.array([[0.3, 0.5, 0.7], [0.9, 0.2, 0.6]], dtype=np.float32),
        "input2": np.array([[0.1, 0.4, 0.9], [0.2, 0.7, 0.5]], dtype=np.float32),
        "target": np.array([1, -1], dtype=np.float32),
        "margin": 0.2,
        "reduction": "mean"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()