import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"]).long()
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

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        target_tensor = tf.convert_to_tensor(input_dict["target"], dtype=tf.int32)
        weight = input_dict.get("weight", None)
        if weight is not None:
            weight = tf.convert_to_tensor(weight, dtype=tf.float32)
        ignore_index = input_dict.get("ignore_index", -100)
        reduction = input_dict.get("reduction", 'mean')
        label_smoothing = input_dict.get("label_smoothing", 0.0)

        if label_smoothing > 0.0:
            num_classes = input_tensor.shape[-1]
            one_hot_target = tf.one_hot(target_tensor, depth=num_classes)
            target_tensor = one_hot_target * (1 - label_smoothing) + label_smoothing / num_classes
        else:
            target_tensor = tf.one_hot(target_tensor, depth=input_tensor.shape[-1])

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_tensor, logits=input_tensor)

        if weight is not None:
            loss = loss * tf.gather(weight, input_dict["target"])

        if reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)
        else:
            pass

        result = loss.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32),
        "target": np.array([2, 0], dtype=np.int64),
        "weight": np.array([0.2, 0.3, 0.5], dtype=np.float32),
        "ignore_index": -100,
        "reduction": 'mean',
        "label_smoothing": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()