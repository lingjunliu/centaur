import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target_tensor = torch.tensor(input_dict["target"])
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
    
    loss = torch.nn.MultiLabelMarginLoss(reduction=reduction)
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
        target_tensor = tf.cast(tf.constant(input_dict["target"]), dtype=tf.float32)
        reduction = input_dict.get("reduction", 'mean')

        def multilabel_margin_loss(logits, labels, reduction='mean'):
            mask = tf.abs(labels)
            num_labels = tf.reduce_sum(mask, axis=1, keepdims=True)

            cost = tf.maximum(0.0, 1.0 - logits * labels)
            loss = tf.reduce_sum(cost, axis=1, keepdims=True) / (num_labels + 1e-8)

            if reduction == 'mean':
                return tf.reduce_mean(loss)
            elif reduction == 'sum':
                return tf.reduce_sum(loss)
            else:
                return loss

        result = multilabel_margin_loss(input_tensor, target_tensor, reduction=reduction)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.6704, 0.4074, 0.4727, 0.9720, 0.1391],
                           [0.4230, 0.6398, 0.3793, 0.9492, 0.6796]], dtype=np.float32),
        "target": np.array([[0, -1, 1, 0, 0],
                            [1, 1, 0, 0, -1]], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.8013, 0.2750, 0.3582, 0.6349, 0.4861],
                           [0.7367, 0.8497, 0.7224, 0.9971, 0.2780]], dtype=np.float32),
        "target": np.array([[1, 0, 1, 0, 0],
                            [0, 0, 0, 1, -1]], dtype=np.int64),
        "reduction": 'sum'
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()