import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()
    
    loss = torch.nn.BCELoss(weight=weight, reduction=reduction)
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        target_tensor = tf.constant(input_dict["target"], dtype=tf.float32)
        weight = input_dict.get("weight", None)
        if weight is not None:
            weight = tf.constant(weight, dtype=tf.float32)
        size_average = input_dict.get("size_average", None)
        reduce = input_dict.get("reduce", None)
        reduction = input_dict.get("reduction", 'mean')

        if weight is not None:
            bce = tf.keras.losses.binary_crossentropy(target_tensor, input_tensor, from_logits=False)
            weighted_bce = bce * weight
            if reduction == 'none':
                result = weighted_bce
            elif reduction == 'sum':
                result = tf.reduce_sum(weighted_bce)
            else:
                result = tf.reduce_mean(weighted_bce)
        else:
            if reduction == 'none':
                result = tf.keras.losses.binary_crossentropy(target_tensor, input_tensor, from_logits=False)
            elif reduction == 'sum':
                result = tf.reduce_sum(tf.keras.losses.binary_crossentropy(target_tensor, input_tensor, from_logits=False))
            else:
                result = tf.reduce_mean(tf.keras.losses.binary_crossentropy(target_tensor, input_tensor, from_logits=False))
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.9, 0.2, 0.7, 0.3], dtype=np.float32),
        "target": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "weight": np.array([0.5, 1.0, 0.5, 1.0], dtype=np.float32),
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()