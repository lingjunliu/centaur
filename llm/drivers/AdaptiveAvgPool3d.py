import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.AdaptiveAvgPool3d(output_size)(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    output_size = input_dict["output_size"]
    
    def adaptive_avg_pool3d(input_tensor, output_size):
        shape = tf.shape(input_tensor)
        b, c, d, h, w = shape[0], shape[1], shape[2], shape[3], shape[4]
        output_d, output_h, output_w = output_size

        input_tensor = tf.reshape(input_tensor, [b * c * d, h, w, 1])
        resized = tf.image.resize(input_tensor, size=[output_h, output_w], method='area')
        resized = tf.reshape(resized, [b * c, d, output_h, output_w, 1])
        resized = tf.transpose(resized, perm=[0, 2, 3, 1, 4])
        resized = tf.reshape(resized, [b * c * output_h * output_w, d, 1, 1])
        resized = tf.image.resize(resized, size=[output_d, 1], method='area')
        resized = tf.reshape(resized, [b, c, output_h, output_w, output_d])
        resized = tf.transpose(resized, perm=[0, 1, 4, 2, 3])

        return resized

    result = adaptive_avg_pool3d(input_tensor, output_size)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 8, 8, 8).astype(np.float32),
        "output_size": (4, 4, 4)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()