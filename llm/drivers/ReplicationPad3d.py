import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict.get("padding")
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    replication_pad3d = torch.nn.ReplicationPad3d(padding)
    result = replication_pad3d(input_tensor)
    
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
        padding = input_dict.get("padding")

        if len(padding) == 6:
            pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom = padding
            paddings_tf = [[0, 0], [pad_front, pad_back], [pad_left, pad_right], [pad_top, pad_bottom], [0, 0]]
        elif len(padding) == 3:
            pad_front = pad_back = padding[0]
            pad_left = pad_right = padding[1]
            pad_top = pad_bottom = padding[2]
            paddings_tf = [[0, 0], [pad_front, pad_back], [pad_left, pad_right], [pad_top, pad_bottom], [0, 0]]
        else:
            raise ValueError("Padding must be of length 3 or 6.")

        result = tf.pad(input_tensor, paddings_tf, mode='REFLECT')

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "padding": (1, 0, 1, 0, 1, 0)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()