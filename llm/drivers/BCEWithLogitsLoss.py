import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    pos_weight = input_dict.get("pos_weight", None)
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()
        if pos_weight is not None:
            pos_weight = pos_weight.cuda()
    
    loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction=reduction)
    result = loss_fn(input_tensor, target_tensor)
    
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
        pos_weight = input_dict.get("pos_weight", None)
        reduction = input_dict.get("reduction", 'mean')

        logits = input_tensor
        labels = target_tensor

        if pos_weight is not None:
            pos_weight = tf.constant(pos_weight, dtype=tf.float32)
            log_weight = (pos_weight - 1) * labels + 1
            bce = tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=pos_weight)
        else:
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

        if weight is not None:
            weight = tf.constant(weight, dtype=tf.float32)
            bce = bce * weight
        
        if reduction == 'mean':
            result = tf.reduce_mean(bce)
        elif reduction == 'sum':
            result = tf.reduce_sum(bce)
        else:
            result = bce
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.5, 0.8, -0.2, 1.1], dtype=np.float32),
        "target": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "weight": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        "pos_weight": np.array([2.0], dtype=np.float32),
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()