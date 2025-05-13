import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"], dtype=torch.long)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
    
    result = torch.take(input_tensor, index)
    
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
        index = tf.constant(input_dict["index"])
        
        input_flat = tf.reshape(input_tensor, [-1])
        indices = tf.cast(index, tf.int32)
        result = tf.gather(input_flat, indices)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[4, 3, 5], [6, 7, 8]], dtype=np.int32),
        "index": np.array([0, 2, 5], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()