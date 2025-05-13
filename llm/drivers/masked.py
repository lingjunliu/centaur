import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    mask = torch.tensor(input_dict["mask"], dtype=torch.bool)
    value = input_dict.get("value", float('nan'))

    if not cpu:
        input_tensor = input_tensor.cuda()
        mask = mask.cuda()

    result = torch.masked_fill(input_tensor, mask, value)

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
        mask = tf.constant(input_dict["mask"])
        value = input_dict.get("value", float('nan'))
        
        mask_bool = tf.cast(mask, tf.bool)
        result = tf.where(mask_bool, tf.fill(tf.shape(input_tensor), tf.cast(value, input_tensor.dtype)), input_tensor)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "mask": np.array([True, False, True, False], dtype=bool),
        "value": 0.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()