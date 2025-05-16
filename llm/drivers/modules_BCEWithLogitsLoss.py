import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    pos_weight = input_dict.get("pos_weight", None)
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
        if weight is not None:
            weight = weight.cuda()
        if pos_weight is not None:
            pos_weight = pos_weight.cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction=reduction)
    result = loss_fn(input_tensor, target)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        target = tf.constant(input_dict["target"], dtype=tf.float32)
        weight = input_dict.get("weight", None)
        if weight is not None:
            weight = tf.constant(weight, dtype=tf.float32)
        pos_weight = input_dict.get("pos_weight", None)
        if pos_weight is not None:
            pos_weight = tf.constant(pos_weight, dtype=tf.float32)
        reduction = input_dict.get("reduction", 'mean')

        log_sigmoid = tf.math.log_sigmoid(input_tensor)

        if pos_weight is not None:
            log_sigmoid_pos_weighted = log_sigmoid * pos_weight
        else:
            log_sigmoid_pos_weighted = log_sigmoid

        loss = target * (log_sigmoid_pos_weighted) + (1 - target) * tf.math.log_sigmoid(-input_tensor)
        loss = -loss

        if weight is not None:
            loss = loss * weight

        if reduction == 'none':
            result = loss
        elif reduction == 'mean':
            result = tf.reduce_mean(loss)
        elif reduction == 'sum':
            result = tf.reduce_sum(loss)
        else:
            raise ValueError("Invalid reduction option")

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.5, 1.2, -0.8], dtype=np.float32),
        "target": np.array([1.0, 0.0, 1.0], dtype=np.float32),
        "weight": np.array([0.5, 1.0, 0.25], dtype=np.float32),
        "pos_weight": np.array([2.0, 1.0, 0.5], dtype=np.float32),
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()