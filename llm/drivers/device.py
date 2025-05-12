import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    fill_value = input_dict.get("fill_value", 0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nan_to_num(input_tensor, nan=fill_value)
    
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
        fill_value = input_dict.get("fill_value", 0)
        
        result = tf.where(tf.math.is_nan(input_tensor), fill_value, input_tensor)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([float('nan'), 1.0, 2.0, float('nan')], dtype=np.float32),
        "fill_value": 0.0
    }

    torch_result = torch_version(input_data)
    
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()