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
    if not input_dict.get("training", True):
        m.eval()
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        alpha = input_dict.get("alpha", -1.0)
        
        training = input_dict.get("training", True)
        
        if training:
            input_shape = tf.shape(input_tensor)
            feature_dropout_mask = tf.cast(tf.random.uniform(shape=input_shape) > p, dtype=tf.float32)
            
            scale = (1.0 - p)
            
            dropped = feature_dropout_mask * input_tensor / scale
            result = alpha + dropped * (1 - alpha)
            
            result = tf.where(tf.math.is_finite(result), result, tf.zeros_like(result) + alpha)
        else:
            result = tf.identity(input_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
        "p": 0.3,
        "training": True,
        "alpha": -1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()