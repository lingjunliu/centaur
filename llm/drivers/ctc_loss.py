import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["log_probs"])
    target_tensor = torch.tensor(input_dict["targets"])
    input_lengths_tensor = torch.tensor(input_dict["input_lengths"])
    target_lengths_tensor = torch.tensor(input_dict["target_lengths"])
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        input_lengths_tensor = input_lengths_tensor.cuda()
        target_lengths_tensor = target_lengths_tensor.cuda()

    input_lengths = tuple(input_lengths_tensor.int().tolist())
    target_lengths = tuple(target_lengths_tensor.int().tolist())

    result = torch.ctc_loss(input_tensor, target_tensor, input_lengths, target_lengths, blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    log_probs = tf.constant(input_dict["log_probs"])
    targets = tf.constant(input_dict["targets"], dtype=tf.int32)
    input_lengths = tf.constant(input_dict["input_lengths"], dtype=tf.int32)
    target_lengths = tf.constant(input_dict["target_lengths"], dtype=tf.int32)
    blank = input_dict.get("blank", 0)
    reduction = input_dict.get("reduction", 'mean')
    zero_infinity = input_dict.get("zero_infinity", False)

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    with tf.device(device_string):
        loss = tf.keras.backend.ctc_batch_cost(targets, log_probs, input_lengths, target_lengths)

        if zero_infinity:
            mask = tf.math.reduce_any(tf.math.is_inf(loss), axis=0)
            loss = tf.where(mask, tf.zeros_like(loss), loss)

        if reduction == 'none':
            result = loss
        elif reduction == 'mean':
            result = tf.reduce_mean(loss)
        elif reduction == 'sum':
            result = tf.reduce_sum(loss)
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
        
        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "log_probs": np.array([[[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.6, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.6, 0.1],
                                 [0.6, 0.1, 0.1, 0.1, 0.1]]], dtype=np.float32),
        "targets": np.array([1, 2, 3], dtype=np.int32),
        "input_lengths": np.array([4], dtype=np.int64),
        "target_lengths": np.array([3], dtype=np.int64),
        "blank": 0,
        "reduction": 'mean',
        "zero_infinity": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()