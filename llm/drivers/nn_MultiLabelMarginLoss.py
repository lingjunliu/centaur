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
    
    loss = torch.nn.MultiLabelMarginLoss(size_average=size_average, reduce=reduce, reduction=reduction)
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
        target_tensor = tf.constant(input_dict["target"], dtype=tf.int64)
        reduction = input_dict.get("reduction", 'mean')
        
        batch_size = tf.shape(input_tensor)[0]
        num_classes = tf.shape(input_tensor)[1]
        
        loss_sum = 0.0
        
        for i in tf.range(batch_size):
            x = input_tensor[i]
            y = target_tensor[i]
            
            targets = tf.boolean_mask(y, tf.greater_equal(y, 0))
            num_targets = tf.cast(tf.shape(targets)[0], dtype=tf.int32)
            
            sample_loss = 0.0
            
            for j in tf.range(tf.cast(num_targets, dtype=tf.int32)):
                yj = targets[j]
                yj = tf.cast(yj, dtype=tf.int32)
                
                for k in tf.range(num_classes):
                    k = tf.cast(k, dtype=tf.int32)
                    
                    if not tf.reduce_any(tf.equal(targets, tf.cast(k, dtype=tf.int64))):
                        margin = tf.maximum(0.0, 1.0 - (x[yj] - x[k]))
                        sample_loss += margin

            sample_loss /= tf.cast(num_classes, dtype=tf.float32) if num_classes > 0 else 1.0

            loss_sum += sample_loss
            
        if reduction == 'mean':
            result = loss_sum / tf.cast(batch_size, dtype=tf.float32)
        elif reduction == 'sum':
            result = loss_sum
        else:
            result = loss_sum
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2, 0.4, 0.8]], dtype=np.float32),
        "target": np.array([[3, 0, -1, 1]], dtype=np.int64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()