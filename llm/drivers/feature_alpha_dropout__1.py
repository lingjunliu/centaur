import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)
    alpha = input_dict.get("alpha", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if training:
        dropout_mask = (torch.rand_like(input_tensor) > p).float()
        output = input_tensor * dropout_mask
        mean = torch.mean(output)
        output = (output - mean) / torch.sqrt(torch.var(output) + 1e-5) * torch.sqrt(input_tensor.var() + 1e-5) + input_tensor.mean()
        output = alpha * (output - alpha * (1 - p)) + alpha * (1 - p) 
    else:
        output = input_tensor

    if not cpu:
        output = output.cpu()

    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)
    alpha = input_dict.get("alpha", 1.0)

    if cpu:
        device = '/cpu:0'
    else:
        device = '/gpu:0'

    with tf.device(device):
        if training:
            random_tensor = tf.random.uniform(shape=tf.shape(input_tensor), minval=0., maxval=1., dtype=tf.float32)
            dropout_mask = tf.cast(random_tensor > p, dtype=tf.float32)
            output = input_tensor * dropout_mask
            mean = tf.reduce_mean(output)
            variance = tf.math.reduce_variance(output)
            output = (output - mean) / tf.sqrt(variance + 1e-5) * tf.sqrt(tf.math.reduce_variance(input_tensor) + 1e-5) + tf.reduce_mean(input_tensor)
            output = alpha * (output - alpha * (1 - p)) + alpha * (1 - p)

        else:
            output = input_tensor

    return {"result": output.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "p": 0.5,
        "training": True,
        "alpha": -1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()