import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    target = torch.tensor(input_dict["target"])
    margin = input_dict.get("margin", 0.0)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()

    loss_fn = torch.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
    result = loss_fn(input1, input2, target)

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
        input1 = tf.constant(input_dict["input1"], dtype=tf.float32)
        input2 = tf.constant(input_dict["input2"], dtype=tf.float32)
        target = tf.constant(input_dict["target"], dtype=tf.float32)
        margin = input_dict.get("margin", 0.0)
        reduction = input_dict.get("reduction", 'mean')

        input1_norm = tf.math.l2_normalize(input1, axis=1)
        input2_norm = tf.math.l2_normalize(input2, axis=1)
        similarity = tf.reduce_sum(input1_norm * input2_norm, axis=1)

        loss = tf.where(
            tf.equal(target, 1.0),
            1 - similarity,
            tf.clip_by_value(similarity - margin, clip_value_min=0.0, clip_value_max=tf.float32.max)
        )

        if reduction == 'mean':
            result = tf.reduce_mean(loss)
        elif reduction == 'sum':
            result = tf.reduce_sum(loss)
        else:
            result = loss

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input1": np.array([[0.3, 0.8, 0.5], [0.2, 0.9, 0.3]], dtype=np.float32),
        "input2": np.array([[0.4, 0.1, 0.9], [0.5, 0.8, 0.2]], dtype=np.float32),
        "target": np.array([1, -1], dtype=np.float32),
        "margin": 0.2,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()