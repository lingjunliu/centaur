import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    margin = input_dict.get("margin", 1.0)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
    
    result = torch.nn.HingeEmbeddingLoss(margin=margin, reduction=reduction)(input_tensor, target)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        target = tf.constant(input_dict["target"], dtype=tf.float32)
        margin = input_dict.get("margin", 1.0)
        reduction = input_dict.get("reduction", 'mean')

        target_float = tf.cast(target, dtype=tf.float32)
        loss = tf.maximum(0.0, margin - tf.multiply(input_tensor, target_float))

        if reduction == 'none':
            result = loss
        elif reduction == 'sum':
            result = tf.reduce_sum(loss)
        else:
            result = tf.reduce_mean(loss)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        "target": np.array([1, 1, -1, -1], dtype=np.int64),
        "margin": 1.0,
        "reduction": "mean"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()