import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    inplace = input_dict.get("inplace", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    m = nn.AlphaDropout(p=p, inplace=inplace)
    result = m(input_tensor)
    
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
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        
        keep_prob = 1.0 - p
        alpha = -np.sqrt((1.0 - keep_prob) / (keep_prob * (1.0 - keep_prob)))
        
        random_tensor = tf.random.uniform(shape=tf.shape(input_tensor), dtype=tf.float32)
        
        binary_mask = tf.cast(random_tensor >= p, dtype=tf.float32)
        
        a = (alpha + alpha * tf.random.normal(shape=tf.shape(input_tensor), mean=0.0, stddev=1.0))
        
        result = input_tensor * binary_mask + a * (1 - binary_mask)
        
        var = keep_prob + p * alpha**2
        scale = np.sqrt(var)

        result = result / scale

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.randn(20, 16).astype(np.float32),
        "p": 0.2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()