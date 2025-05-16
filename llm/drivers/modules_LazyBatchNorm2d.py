import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()

    bn = torch.nn.LazyBatchNorm2d()
    if not cpu:
        bn = bn.cuda()

    result = bn(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)

        if len(input_tensor.shape) == 1:
            input_tensor = tf.reshape(input_tensor, (1, 1, 1, input_tensor.shape[0]))
        elif len(input_tensor.shape) == 2:
            input_tensor = tf.reshape(input_tensor, (1, 1, input_tensor.shape[0], input_tensor.shape[1]))
        elif len(input_tensor.shape) == 3:
            input_tensor = tf.reshape(input_tensor, (1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]))

        input_shape = input_tensor.shape
        num_channels = input_shape[-1]

        gamma = tf.Variable(tf.ones(num_channels, dtype=tf.float32))
        beta = tf.Variable(tf.zeros(num_channels, dtype=tf.float32))
        
        axes = [0, 1, 2]
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)

        epsilon = 1e-5
        
        normalized_tensor = (input_tensor - mean) / tf.sqrt(variance + epsilon)

        result = gamma * normalized_tensor + beta

        result = result.numpy()

        gamma_np = np.ones(num_channels, dtype=np.float32)
        beta_np = np.zeros(num_channels, dtype=np.float32)
        mean_np = np.mean(input_dict["input"], axis=(0, 1, 2))
        variance_np = np.var(input_dict["input"], axis=(0, 1, 2))
        epsilon_np = 1e-5
        
        normalized_np = (input_dict["input"] - mean_np) / np.sqrt(variance_np + epsilon_np)
        output_np = gamma_np * normalized_np + beta_np
        result = output_np
        result = tf.reshape(tf.convert_to_tensor(result), input_shape).numpy()


    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()