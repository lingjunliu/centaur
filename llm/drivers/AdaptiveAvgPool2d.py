import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.AdaptiveAvgPool2d(output_size)(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    output_size = input_dict["output_size"]

    if cpu:
        with tf.device("/cpu:0"):
            result = tf.transpose(tf.image.resize(tf.transpose(input_tensor, perm=[0, 2, 3, 1]), output_size, method=tf.image.ResizeMethod.AREA), perm=[0, 3, 1, 2])
    else:
        with tf.device("/gpu:0"):
            result = tf.transpose(tf.image.resize(tf.transpose(input_tensor, perm=[0, 2, 3, 1]), output_size, method=tf.image.ResizeMethod.AREA), perm=[0, 3, 1, 2])
    
    return {"result": result.numpy()}

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