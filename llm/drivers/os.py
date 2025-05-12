import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    q_scale = torch.tensor(input_dict.get("q_scale", 1.0))
    q_zero_point = torch.tensor(input_dict.get("q_zero_point", 0))
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        q_scale = q_scale.cuda()
        q_zero_point = q_zero_point.cuda()
    
    result = input_tensor * q_scale * (2 ** q_zero_point)
    
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
        input_tensor = tf.constant(input_dict["input"])
        q_scale = tf.constant(input_dict.get("q_scale", 1.0))
        q_zero_point = tf.constant(input_dict.get("q_zero_point", 0))
        
        result = tf.cast(input_tensor, tf.float32) * tf.cast(q_scale, tf.float32) * (2 ** tf.cast(q_zero_point, tf.float32))
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.int32),
        "q_scale": 2.0,
        "q_zero_point": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()