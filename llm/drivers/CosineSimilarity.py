import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    dim = input_dict.get("dim", 1)
    eps = input_dict.get("eps", 1e-08)

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()
    
    cos = torch.nn.CosineSimilarity(dim=dim, eps=eps)
    result = cos(input1, input2)
    
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
        dim = input_dict.get("dim", 1)
        eps = input_dict.get("eps", 1e-08)
        
        input1_norm = tf.linalg.norm(input1, axis=dim, keepdims=True)
        input2_norm = tf.linalg.norm(input2, axis=dim, keepdims=True)
        
        numerator = tf.reduce_sum(tf.multiply(input1, input2), axis=dim, keepdims=True)
        denominator = tf.multiply(input1_norm, input2_norm)
        
        result = numerator / (denominator + eps)
        result = tf.squeeze(result, axis=dim)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input1": np.array([[0.0202, 1.0985, 1.3506, -0.6056], [0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "input2": np.array([[0.0234, 1.0512, 1.3473, -0.6234], [0.0234, 1.0512, 1.3473, -0.6234]], dtype=np.float32),
        "dim": 1,
        "eps": 1e-08
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()