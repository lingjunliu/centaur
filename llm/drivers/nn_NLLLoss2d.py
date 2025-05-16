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
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    ignore_index = input_dict.get("ignore_index", -100)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()

    loss = torch.nn.NLLLoss2d(weight=weight, ignore_index=ignore_index, reduction=reduction)
    result = loss(input_tensor, target_tensor)

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
        target_tensor = tf.constant(input_dict["target"], dtype=tf.int32)
        weight = input_dict.get("weight", None)
        if weight is not None:
            weight = tf.constant(weight)
        ignore_index = input_dict.get("ignore_index", -100)
        reduction = input_dict.get("reduction", 'mean')

        input_shape = tf.shape(input_tensor)
        n_samples = tf.cast(input_shape[0], tf.float32)
        height = tf.cast(input_shape[2], tf.float32)
        width = tf.cast(input_shape[3], tf.float32)

        batch_size = tf.shape(input_tensor)[0]
        channels = tf.shape(input_tensor)[1]
        height = tf.shape(input_tensor)[2]
        width = tf.shape(input_tensor)[3]

        target_one_hot = tf.one_hot(target_tensor, depth=channels)

        log_probs = input_tensor
        
        # Apply mask for ignore_index
        mask = tf.cast(tf.not_equal(target_tensor, ignore_index), tf.float32)

        # Calculate the loss for each pixel
        per_pixel_loss = -tf.reduce_sum(target_one_hot * log_probs, axis=1) * mask

        if weight is not None:
            per_pixel_loss = per_pixel_loss * tf.gather(weight, target_tensor)

        # Apply reduction
        if reduction == 'mean':
            loss = tf.reduce_sum(per_pixel_loss) / tf.reduce_sum(mask)
        elif reduction == 'sum':
            loss = tf.reduce_sum(per_pixel_loss)
        else:  # reduction == 'none'
            loss = per_pixel_loss

        result = loss.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]],
                            [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]]]], dtype=np.float32),
        "target": np.array([[[0, 1], [2, 0]]]], dtype=np.int64),
        "weight": np.array([0.2, 0.3, 0.5], dtype=np.float32),
        "ignore_index": -100,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()