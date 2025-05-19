import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    ignore_index = input_dict.get("ignore_index", -100)
    reduction = input_dict.get("reduction", "mean")
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
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        target_tensor = tf.constant(input_dict["target"])
        weight = input_dict.get("weight", None)
        if weight is not None:
            weight = tf.constant(weight)
        ignore_index = input_dict.get("ignore_index", -100)
        reduction = input_dict.get("reduction", "mean")
        label_smoothing = input_dict.get("label_smoothing", 0.0)

        if label_smoothing > 0.0:
            num_classes = input_tensor.shape[-1]
            target_tensor = tf.one_hot(target_tensor, depth=num_classes)
            target_tensor = target_tensor * (1 - label_smoothing) + label_smoothing / num_classes

        if len(input_tensor.shape) > 2:
            input_tensor = tf.reshape(input_tensor, (-1, input_tensor.shape[-1]))
            target_tensor = tf.reshape(target_tensor, (-1,))

        log_probs = tf.nn.log_softmax(input_tensor, axis=-1)
        
        if label_smoothing > 0.0:
            per_example_loss = -tf.reduce_sum(target_tensor * log_probs, axis=-1)
        else:
            per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(target_tensor, tf.int32), logits=input_tensor)
            
        if ignore_index != -100:
             mask = tf.not_equal(target_tensor, ignore_index)
             per_example_loss = tf.boolean_mask(per_example_loss, mask)

        if weight is not None:
             if label_smoothing > 0.0:
                per_example_loss = per_example_loss
             else:
                weights = tf.gather(weight, target_tensor)
                per_example_loss = per_example_loss * weights


        if reduction == "mean":
            result = tf.reduce_mean(per_example_loss)
        elif reduction == "sum":
            result = tf.reduce_sum(per_example_loss)
        else:
            result = per_example_loss
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2, 0.7], [0.2, 0.8, 0.0]], dtype=np.float32),
        "target": np.array([2, 0], dtype=np.int64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.1, 0.2, 0.7], [0.2, 0.8, 0.0]], dtype=np.float32),
        "target": np.array([2, 0], dtype=np.int64),
        "weight": np.array([0.2, 0.3, 0.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.1, 0.2, 0.7], [0.2, 0.8, 0.0]], dtype=np.float32),
        "target": np.array([2, 0], dtype=np.int64),
        "ignore_index": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.1, 0.2, 0.7], [0.2, 0.8, 0.0]], dtype=np.float32),
        "target": np.array([2, 0], dtype=np.int64),
        "reduction": "sum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.1, 0.2, 0.7], [0.2, 0.8, 0.0]], dtype=np.float32),
        "target": np.array([2, 0], dtype=np.int64),
        "label_smoothing": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()