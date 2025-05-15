import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    p = input_dict.get("p", 1)
    margin = input_dict.get("margin", 1.0)
    weight = input_dict.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    size_average = input_dict.get("size_average", None)
    reduce = input_dict.get("reduce", None)
    reduction = input_dict.get("reduction", 'mean')

    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
        if weight is not None:
            weight = weight.cuda()

    loss_fn = torch.nn.MultiMarginLoss(p=p, margin=margin, weight=weight, reduction=reduction)
    result = loss_fn(input_tensor, target)

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
        target = tf.constant(input_dict["target"], dtype=tf.int32)
        p = input_dict.get("p", 1)
        margin = input_dict.get("margin", 1.0)
        weight = input_dict.get("weight", None)
        reduction = input_dict.get("reduction", 'mean')
        
        def multi_margin_loss(inputs, target, p=1, margin=1.0, weights=None, reduction='mean'):
            batch_size = tf.shape(inputs)[0]
            num_classes = tf.shape(inputs)[1]

            losses = tf.zeros([batch_size], dtype=tf.float32)

            for i in range(batch_size):
                correct_class_score = inputs[i, target[i]]
                
                sample_loss = 0.0
                for j in range(num_classes):
                    if j != target[i]:
                        loss = margin - correct_class_score + inputs[i, j]
                        loss = tf.maximum(loss, 0.0)
                        if weights is not None:
                            sample_loss += loss * weights[j]
                        else:
                            sample_loss += loss
                
                losses = tf.tensor_scatter_nd_update(losses, [[i]], [sample_loss])
            
            if reduction == 'mean':
                loss = tf.reduce_mean(losses) / tf.cast((num_classes - 1), tf.float32)
            elif reduction == 'sum':
                loss = tf.reduce_sum(losses) / tf.cast((num_classes - 1), tf.float32)
            else:
                loss = losses / tf.cast((num_classes - 1), tf.float32)
            return loss
        
        if weight is not None:
            result = multi_margin_loss(input_tensor, target, p=p, margin=margin, weights=weight, reduction=reduction)
        else:
            result = multi_margin_loss(input_tensor, target, p=p, margin=margin, reduction=reduction)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
        "target": np.array([0, 1], dtype=np.int64),
        "p": 1,
        "margin": 0.5,
        "weight": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()