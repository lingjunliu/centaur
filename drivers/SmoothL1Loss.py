import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    beta = input_dict.get("beta", 1.0)
    reduction = input_dict.get("reduction", 'mean')
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
    
    loss = torch.nn.SmoothL1Loss(reduction=reduction, beta=beta)
    result = loss(input_tensor, target)
    
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
        beta = input_dict.get("beta", 1.0)
        reduction = input_dict.get("reduction", 'mean')
        
        abs_diff = tf.abs(input_tensor - target)
        
        loss = tf.cond(
            tf.reduce_any(abs_diff < beta),
            lambda: tf.where(
                abs_diff < beta,
                0.5 * (abs_diff ** 2) / beta,
                abs_diff - 0.5 * beta
            ),
            lambda: abs_diff - 0.5 * beta
        )

        if reduction == 'mean':
            result = tf.reduce_mean(loss)
        elif reduction == 'sum':
            result = tf.reduce_sum(loss)
        else:
            result = loss
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "target": np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "beta": 1.0,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()