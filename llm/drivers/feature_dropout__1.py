import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)

    if not cpu:
        input_tensor = input_tensor.cuda()

    mask = (torch.rand(input_tensor.shape, device=input_tensor.device) >= p).float()

    output_tensor = input_tensor * mask
    output_tensor = output_tensor / (1 - p)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"result": output_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)

        mask = tf.cast(tf.random.uniform(shape=tf.shape(input_tensor), dtype=tf.float32) >= p, dtype=tf.float32)
        output_tensor = tf.multiply(input_tensor, mask)
        output_tensor = output_tensor / (1 - p)

    return {"result": output_tensor.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "p": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()