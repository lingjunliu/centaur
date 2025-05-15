import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    norm = torch.nn.LazyInstanceNorm2d()
    result = norm(input_tensor)

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
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        input_shape = input_tensor.shape
        rank = len(input_shape)

        if rank == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=[0, 1])
        elif rank == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        elif rank == 4:
            pass
        else:
            raise ValueError("Input tensor must have rank 2, 3, or 4")

        num_channels = input_tensor.shape[-1]
        num_instances = input_tensor.shape[0] * input_tensor.shape[1] if rank > 3 else input_tensor.shape[0]

        def instance_norm(input_tensor, epsilon=1e-5):
            axes = [1, 2]
            mean = tf.math.reduce_mean(input_tensor, axis=axes, keepdims=True)
            variance = tf.math.reduce_variance(input_tensor, axis=axes, keepdims=True)
            normalized = (input_tensor - mean) / tf.sqrt(variance + epsilon)

            gamma = tf.Variable(tf.ones([num_channels]), dtype=tf.float32)
            beta = tf.Variable(tf.zeros([num_channels]), dtype=tf.float32)

            gamma = tf.reshape(gamma, (1, 1, 1, num_channels))
            beta = tf.reshape(beta, (1, 1, 1, num_channels))

            return gamma * normalized + beta

        result = instance_norm(input_tensor)

        if rank == 2:
            result = tf.squeeze(result, axis=[0, 1])
        elif rank == 3:
            result = tf.squeeze(result, axis=0)

        result_numpy = result.numpy()

    return {"result": result_numpy}

def main():
    A_TOL = 1e-5

    input_data = {
        "input": np.random.rand(2, 3, 64, 64).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()