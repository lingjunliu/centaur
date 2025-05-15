import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(output_size)

    if not cpu:
        adaptive_avg_pool2d = adaptive_avg_pool2d.cuda()

    result = adaptive_avg_pool2d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    output_size = input_dict["output_size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = input_tensor.shape
        ksize = [1, input_shape[1] // output_size[0], input_shape[2] // output_size[1], 1]
        strides = [1, input_shape[1] // output_size[0], input_shape[2] // output_size[1], 1]
        padding = 'VALID'
        
        result = tf.nn.avg_pool(input_tensor, ksize=ksize, strides=strides, padding=padding)
        
        result = tf.image.resize(result, size=output_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 8, 8, 3).astype(np.float32),
        "output_size": (4, 4)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()