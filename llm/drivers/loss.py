import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    ignore_index = input_dict.get("ignore_index", -100)
    reduction = input_dict.get("reduction", 'mean')
    label_smoothing = input_dict.get("label_smoothing", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()

    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
    result = loss_fn(input_tensor, target_tensor)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        target_tensor = tf.constant(input_dict["target"])
        weight = input_dict.get("weight", None)
        ignore_index = input_dict.get("ignore_index", -100)
        reduction = input_dict.get("reduction", 'mean')
        label_smoothing = input_dict.get("label_smoothing", 0.0)

        num_classes = input_tensor.shape[-1]
        target_one_hot = tf.one_hot(target_tensor, depth=num_classes, dtype=tf.float32)
        if label_smoothing > 0.0:
            target_smoothed = (1.0 - label_smoothing) * target_one_hot + label_smoothing / num_classes
        else:
            target_smoothed = target_one_hot

        log_probs = tf.nn.log_softmax(input_tensor, axis=-1)
        loss = -tf.reduce_sum(target_smoothed * log_probs, axis=-1)

        if weight is not None:
            weight_tensor = tf.constant(weight, dtype=tf.float32)
            loss = loss * tf.gather(weight_tensor, target_tensor)

        if ignore_index != -100:
            mask = tf.cast(tf.not_equal(target_tensor, ignore_index), dtype=tf.float32)
            loss = loss * mask

        if reduction == 'sum':
            loss = tf.reduce_sum(loss)
        elif reduction == 'mean':
            if ignore_index != -100:
                valid_count = tf.reduce_sum(mask)
                loss = tf.reduce_sum(loss) / valid_count if valid_count > 0 else tf.constant(0.0, dtype=tf.float32)
            else:
                loss = tf.reduce_mean(loss)

        result = loss.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], dtype=np.float32),
        "target": np.array([2, 0], dtype=np.int64),
        "weight": np.array([0.2, 0.3, 0.5], dtype=np.float32),
        "ignore_index": 0,
        "reduction": 'mean',
        "label_smoothing": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()