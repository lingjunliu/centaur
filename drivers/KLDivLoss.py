import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    reduction = input_dict.get("reduction", 'mean')
    log_target = input_dict.get("log_target", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

    loss = torch.nn.KLDivLoss(reduction=reduction, log_target=log_target)
    result = loss(input_tensor.log_softmax(dim=-1), target_tensor)

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
        target_tensor = tf.constant(input_dict["target"], dtype=tf.float32)
        reduction = input_dict.get("reduction", 'mean')
        log_target = input_dict.get("log_target", False)

        log_prob = tf.nn.log_softmax(input_tensor, axis=-1)
        
        if not log_target:
            kl_div = target_tensor * (tf.math.log(target_tensor) - log_prob)
        else:
            kl_div = target_tensor * (- log_prob)

        if reduction == 'none':
            result = kl_div
        elif reduction == 'sum':
            result = tf.reduce_sum(kl_div)
        elif reduction == 'batchmean':
            result = tf.reduce_mean(kl_div)
        elif reduction == 'mean':
            result = tf.reduce_mean(tf.reduce_sum(kl_div, axis=-1))
        else:
            result = tf.reduce_sum(kl_div) / tf.cast(tf.size(input_tensor), dtype=tf.float32)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "target": np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32),
        "reduction": 'mean',
        "log_target": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "target": np.array([-2.3026, -0.9163, -1.2040, -1.6094], dtype=np.float32),
        "reduction": 'mean',
        "log_target": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "target": np.array([-2.3026, -0.9163, -1.2040, -1.6094], dtype=np.float32),
        "reduction": 'batchmean',
        "log_target": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "target": np.array([[-2.3026, -0.9163, -1.2040, -1.6094]], dtype=np.float32),
        "reduction": 'mean',
        "log_target": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()