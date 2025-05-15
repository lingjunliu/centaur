import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    inplace = input_dict.get("inplace", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    m = nn.FeatureAlphaDropout(p=p, inplace=inplace)
    result = m(input_tensor)
    
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
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        
        input_shape = tf.shape(input_tensor)
        rank = len(input_tensor.shape)
        
        if rank not in (4, 5):
            raise ValueError("Input tensor must have rank 4 or 5.")
            
        neg_saturation = -1.7580
        alpha_val = tf.cast(-np.sqrt(2.0 / ((1 - p) + p * neg_saturation**2)), dtype=tf.float32)
        beta_val = tf.cast(-alpha_val * p * neg_saturation, dtype=tf.float32)


        if rank == 5:
            mask_shape = (1, input_shape[1], 1, 1, 1)
            random_tensor = tf.random.uniform(mask_shape, dtype=tf.float32)
            mask = tf.cast(random_tensor > p, dtype=tf.float32)
            x = input_tensor * mask
            result = alpha_val * x + beta_val * (1 - mask)

        else:
            mask_shape = (1, input_shape[0], 1, 1)
            random_tensor = tf.random.uniform(mask_shape, dtype=tf.float32)
            mask = tf.cast(random_tensor > p, dtype=tf.float32)
            x = input_tensor * mask
            result = alpha_val * x + beta_val * (1 - mask)
            
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(20, 16, 4, 32, 32).astype(np.float32),
        "p": 0.2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()