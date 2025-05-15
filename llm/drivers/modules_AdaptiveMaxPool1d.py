import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"]).unsqueeze(0).unsqueeze(0)
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.AdaptiveMaxPool1d(output_size)(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    output_size = input_dict["output_size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = tf.shape(input_tensor)
        input_tensor = tf.reshape(input_tensor, (1, input_shape[0], 1))
        
        kernel_size = input_shape[1] // output_size
        if kernel_size == 0:
            kernel_size = 1

        result = tf.nn.max_pool(input_tensor, ksize=[1, kernel_size, 1], strides=[1, input_shape[1] // output_size if input_shape[1] >= output_size else 1, 1], padding='VALID')
        result = tf.reshape(result, (output_size,)).numpy()

    return {"result": result}

def main():
    A_TOL = 0.1

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "output_size": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()