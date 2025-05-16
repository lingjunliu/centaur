import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    target = torch.tensor(input_dict["target"])
    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    margin = input_dict.get("margin", 0.0)
    reduction = input_dict.get("reduction", 'mean')
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        target = target.cuda()
        input1 = input1.cuda()
        input2 = input2.cuda()
    
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
    result = loss_fn(input1, input2, target)
    
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
        input1 = tf.constant(input_dict["input1"], dtype=tf.float32)
        input2 = tf.constant(input_dict["input2"], dtype=tf.float32)
        target = tf.cast(tf.constant(input_dict["target"]), dtype=tf.float32)
        margin = input_dict.get("margin", 0.0)
        reduction = input_dict.get("reduction", 'mean')

        def cosine_similarity(x1, x2):
            x1_norm = tf.norm(x1, axis=-1)
            x2_norm = tf.norm(x2, axis=-1)
            x1 = x1 / (x1_norm + 1e-8)
            x2 = x2 / (x2_norm + 1e-8)
            return tf.reduce_sum(x1 * x2, axis=-1)
        
        similarity = cosine_similarity(input1, input2)
        
        loss = target * (1 - similarity) + (1 - target) * tf.maximum(0.0, similarity + margin)

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
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "target": np.array([1, -1, 1, -1], dtype=np.float32),
        "input1": np.array([[0.3, 0.8, 0.5], [0.9, 0.4, 0.2], [0.2, 0.9, 0.3], [0.7, 0.1, 0.6]], dtype=np.float32),
        "input2": np.array([[0.1, 0.7, 0.2], [0.5, 0.2, 0.8], [0.4, 0.6, 0.1], [0.9, 0.5, 0.3]], dtype=np.float32),
        "margin": 0.5,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()