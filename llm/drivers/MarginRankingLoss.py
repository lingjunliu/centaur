import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor1 = torch.tensor(input_dict["input1"])
    input_tensor2 = torch.tensor(input_dict["input2"])
    target = torch.tensor(input_dict["target"])
    margin = input_dict.get("margin", 0.0)
    reduction = input_dict.get("reduction", 'mean')
    
    if not cpu:
        input_tensor1 = input_tensor1.cuda()
        input_tensor2 = input_tensor2.cuda()
        target = target.cuda()
        
    loss = torch.nn.MarginRankingLoss(margin=margin, reduction=reduction)
    result = loss(input_tensor1, input_tensor2, target)
    
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
        target = tf.constant(input_dict["target"], dtype=tf.float32)
        margin = input_dict.get("margin", 0.0)
        reduction = input_dict.get("reduction", 'mean')

        diff = input1 - input2
        margin_term = tf.maximum(0.0, margin - diff * target)

        if reduction == 'mean':
            result = tf.reduce_mean(margin_term)
        elif reduction == 'sum':
            result = tf.reduce_sum(margin_term)
        else:
            result = margin_term

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "input2": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "target": np.array([1, -1, 1], dtype=np.int64),
        "margin": 0.5,
        "reduction": 'mean'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "input2": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "target": np.array([1, -1, 1], dtype=np.int64),
        "margin": 0.5,
        "reduction": 'sum'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "input2": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "target": np.array([1, -1, 1], dtype=np.int64),
        "margin": 0.5,
        "reduction": 'none'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()