import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    lazy_batch_norm = torch.nn.LazyBatchNorm1d()
    if not cpu:
        lazy_batch_norm = lazy_batch_norm.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    result = lazy_batch_norm(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze(0).squeeze(0).detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        mean, variance = tf.nn.moments(input_tensor, axes=[2])

        epsilon = 1e-5
        normalized_tensor = (input_tensor - mean) / tf.sqrt(variance + epsilon)

        gamma = tf.Variable(tf.ones([input_tensor.shape[-1]], dtype=tf.float32))
        beta = tf.Variable(tf.zeros([input_tensor.shape[-1]], dtype=tf.float32))

        result = gamma * normalized_tensor + beta
        result = tf.squeeze(result, axis=[0, 1])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()