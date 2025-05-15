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

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()

    loss = torch.nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    result = loss(input_tensor, target_tensor)

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
        input_tensor = tf.constant(input_dict["input"])
        target_tensor = tf.constant(input_dict["target"])
        weight = input_dict.get("weight", None)
        if weight is not None:
            weight = tf.constant(weight)
        ignore_index = input_dict.get("ignore_index", -100)
        reduction = input_dict.get("reduction", 'mean')

        input_shape = tf.shape(input_tensor)
        N = input_shape[0]
        C = input_shape[1]
        H = input_shape[2]
        W = input_shape[3]

        input_reshaped = tf.reshape(input_tensor, [N, C, H * W])
        input_transposed = tf.transpose(input_reshaped, perm=[0, 2, 1])
        input_log_probs = tf.reshape(input_transposed, [-1, C])

        target_reshaped = tf.reshape(target_tensor, [-1])

        mask = tf.not_equal(target_reshaped, ignore_index)
        masked_target = tf.boolean_mask(target_reshaped, mask)
        masked_log_probs = tf.boolean_mask(input_log_probs, mask)
        
        indices = tf.stack([tf.range(tf.shape(masked_target)[0]), tf.cast(masked_target, tf.int32)], axis=1)
        
        gathered_log_probs = -tf.gather_nd(masked_log_probs, indices)

        if weight is not None:
            gathered_weights = tf.gather(weight, tf.cast(masked_target, tf.int32))
            weighted_log_probs = gathered_log_probs * gathered_weights
        else:
            weighted_log_probs = gathered_log_probs

        if reduction == 'mean':
            result = tf.reduce_mean(weighted_log_probs)
        elif reduction == 'sum':
            result = tf.reduce_sum(weighted_log_probs)
        else:
            result = weighted_log_probs

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[-0.7243, -0.7769],
                            [-0.7564, -0.7749]],

                           [[-0.6793, -0.8058],
                            [-0.7080, -0.7608]]]], dtype=np.float32),
        "target": np.array([[[0, 1],
                            [1, 0]]], dtype=np.int64),
        "weight": np.array([0.8, 0.2], dtype=np.float32),
        "ignore_index": -100,
        "reduction": 'mean'
    }

    input_data["input"] = np.transpose(input_data["input"], (0, 2, 3, 1))
    torch_input = np.transpose(input_data["input"], (0, 3, 1, 2))
    input_data["input"] = torch_input

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()