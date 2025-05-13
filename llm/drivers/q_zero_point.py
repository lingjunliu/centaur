import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    q_min = torch.iinfo(torch.int8).min
    q_max = torch.iinfo(torch.int8).max

    scale = (input_tensor.max() - input_tensor.min()) / (q_max - q_min)
    zero_point = q_min - torch.round(input_tensor.min() / scale)

    result = torch.tensor(zero_point, dtype=torch.int64)
    
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
        
        q_min = np.iinfo(np.int8).min
        q_max = np.iinfo(np.int8).max

        scale = (tf.reduce_max(input_tensor) - tf.reduce_min(input_tensor)) / (q_max - q_min)
        zero_point = q_min - tf.round(tf.reduce_min(input_tensor) / scale)

        result = tf.cast(zero_point, tf.int64)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 1.0
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()