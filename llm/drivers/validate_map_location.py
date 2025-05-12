import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_map_location = torch.tensor(input_dict["input_map_location"])
    storage = torch.tensor(input_dict["storage"])
    
    if not cpu:
        input_map_location = input_map_location.cuda()
        storage = storage.cuda()
    
    result = torch.zeros_like(input_map_location)
    
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
        input_map_location = tf.constant(input_dict["input_map_location"])
        storage = tf.constant(input_dict["storage"])

        result = tf.zeros_like(input_map_location)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input_map_location": np.array([1, 2, 3], dtype=np.int64),
        "storage": np.array([4, 5, 6], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()