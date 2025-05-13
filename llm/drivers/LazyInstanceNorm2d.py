import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.LazyInstanceNorm2d()(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])

        shape = tf.shape(input_tensor)
        rank = tf.rank(input_tensor)

        if rank == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=1)
        elif rank == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=1)

        mean, variance = tf.nn.moments(input_tensor, axes=[2, 3], keepdims=True)
        epsilon = 1e-5
        result = (input_tensor - mean) / tf.sqrt(variance + epsilon)

        if rank == 2:
            result = tf.squeeze(result, axis=[0, 1])
        elif rank == 3:
            result = tf.squeeze(result, axis=1)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 4, 32, 32).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()