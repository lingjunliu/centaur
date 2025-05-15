import numpy as np
import torch.nn.functional as F

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    log_target = input_dict.get("log_target", False)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()

    if reduction == 'mean':
        reduction_mode = 'mean'
    elif reduction == 'sum':
        reduction_mode = 'sum'
    elif reduction == 'batchmean':
        reduction_mode = 'batchmean'
    else:
        reduction_mode = 'none'

    if log_target:
        result = F.kl_div(input_tensor, target, log_target=log_target, reduction=reduction_mode)
    else:
        result = F.kl_div(F.log_softmax(input_tensor, dim=-1), target, log_target=log_target, reduction=reduction_mode)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        target = tf.constant(input_dict["target"], dtype=tf.float32)
        log_target = input_dict.get("log_target", False)
        reduction = input_dict.get("reduction", 'mean')

        if log_target:
            kl_divergence = tf.reduce_sum(target * (target - input_tensor), axis=-1)
        else:
            log_probs = tf.nn.log_softmax(input_tensor)
            kl_divergence = tf.reduce_sum(target * (tf.math.log(target) - log_probs), axis=-1)
        
        if reduction == 'mean':
            result = tf.reduce_mean(kl_divergence)
        elif reduction == 'sum':
            result = tf.reduce_sum(kl_divergence)
        elif reduction == 'batchmean':
            result = tf.reduce_sum(kl_divergence) / tf.cast(tf.shape(input_tensor)[0], tf.float32)
        else:
            result = kl_divergence

        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.1, 0.2, 0.7], dtype=np.float32),
        "target": np.array([0.3, 0.4, 0.3], dtype=np.float32),
        "log_target": False,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()