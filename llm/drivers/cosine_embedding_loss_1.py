import numpy as np
import torch
import torch.nn as nn

def torch_version(input_dict, cpu=True):
    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    target = torch.tensor(input_dict["target"])
    margin = input_dict.get("margin", 0.0)
    reduction_str = input_dict.get("reduction", 'mean')
    
    if reduction_str == 'mean':
        reduction = 1
    elif reduction_str == 'sum':
        reduction = 0
    else:
        reduction = 2

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()
    
    result = torch.cosine_embedding_loss(input1, input2, target, margin=margin, reduction=reduction)

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
        input1 = tf.constant(input_dict["input1"], dtype=tf.float32)
        input2 = tf.constant(input_dict["input2"], dtype=tf.float32)
        target = tf.cast(tf.constant(input_dict["target"]), dtype=tf.float32)
        margin = input_dict.get("margin", 0.0)
        reduction_str = input_dict.get("reduction", 'mean')

        norm_input1 = tf.linalg.norm(input1, axis=-1)
        norm_input2 = tf.linalg.norm(input2, axis=-1)
        
        similarity = tf.reduce_sum(input1 * input2, axis=-1) / (norm_input1 * norm_input2 + 1e-8)
        
        loss = target * (1 - similarity) + (1 - target) * tf.maximum(0.0, similarity - margin)

        if reduction_str == 'mean':
            result = tf.reduce_mean(loss)
        elif reduction_str == 'sum':
            result = tf.reduce_sum(loss)
        else:
            result = loss

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input1": np.array([[0.3, 0.5, 0.7], [0.8, 0.2, 0.9]], dtype=np.float32),
        "input2": np.array([[0.1, 0.9, 0.2], [0.6, 0.4, 0.5]], dtype=np.float32),
        "target": np.array([1, -1], dtype=np.float32),
        "margin": 0.2,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()