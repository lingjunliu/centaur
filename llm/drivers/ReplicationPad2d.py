import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    replication_pad = torch.nn.ReplicationPad2d(padding)
    result = replication_pad(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        
        if isinstance(padding, int):
            pad_width = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
        elif isinstance(padding, tuple) and len(padding) == 2:
            pad_width = [[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]]
        elif isinstance(padding, tuple) and len(padding) == 4:
            pad_width = [[0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [0, 0]]
            
        else:
            raise ValueError("Padding must be an int or a tuple of length 2 or 4")
        
        input_shape = input_tensor.shape

        if isinstance(padding, tuple) and len(padding) == 4:
            pad_width = [[0, 0], [min(padding[0], input_shape[1]), min(padding[1], input_shape[1])], [min(padding[2], input_shape[2]), min(padding[3], input_shape[2])], [0, 0]]
        else:
            pad_width = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
        
        result = tf.pad(input_tensor, pad_width, mode='REFLECT')
            
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
        "padding": (0, 0, 0, 0)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()