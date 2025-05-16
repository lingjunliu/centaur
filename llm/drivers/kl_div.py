import numpy as np


def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch import nn
    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    log_target = input_dict.get("log_target", False)
    reduction = input_dict.get("reduction", 'mean')

    if reduction == 'none':
        reduction_enum = 'none'
    elif reduction == 'sum':
        reduction_enum = 'sum'
    elif reduction == 'batchmean':
        reduction_enum = 'batchmean'
    else:
        reduction_enum = 'mean'

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()

    result = torch.nn.functional.kl_div(input_tensor.log_softmax(dim=-1), target, log_target=log_target, reduction=reduction_enum)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import tensorflow.keras.backend as K

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        target = tf.constant(input_dict["target"], dtype=tf.float32)
        log_target = input_dict.get("log_target", False)
        reduction = input_dict.get("reduction", 'mean')

        log_prob = tf.nn.log_softmax(input_tensor, axis=-1)

        if not log_target:
            log_target_tf = tf.math.log(target)
        else:
            log_target_tf = target

        kl_divergence = target * (log_target_tf - log_prob)

        if reduction == 'none':
            result = tf.reduce_sum(kl_divergence, axis=-1).numpy()
        elif reduction == 'sum':
            result = tf.reduce_sum(kl_divergence).numpy()
        elif reduction == 'batchmean':
            batch_size = tf.cast(tf.shape(input_tensor)[0], dtype=tf.float32)
            result = (tf.reduce_sum(kl_divergence) / batch_size).numpy()
        else:
            result = tf.reduce_mean(kl_divergence).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "target": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        "log_target": False,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "target": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        "log_target": False,
        "reduction": 'sum'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "target": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        "log_target": False,
        "reduction": 'none'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "target": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        "log_target": True,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()