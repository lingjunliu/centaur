import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    train = input_dict.get("train", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nn.functional.dropout(input_tensor, p=p, training=train)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    p = input_dict.get("p", 0.5)
    train = input_dict.get("train", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        if train:
            result = tf.nn.dropout(input_tensor, rate=p)
        else:
            result = input_tensor

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.1

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "p": 0.5,
        "train": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()