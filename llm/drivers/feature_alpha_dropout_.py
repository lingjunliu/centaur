import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    alpha = input_dict.get("alpha", 1.0)
    training = input_dict.get("training", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if training:
        bernoulli = torch.distributions.bernoulli.Bernoulli(1 - p)
        mask = bernoulli.sample(input_tensor.shape)
        if not cpu:
            mask = mask.cuda()
        mask = mask.float()
        output = input_tensor * mask + alpha * (1 - mask)
        output = output * (1 / (1 - p))
    else:
        output = input_tensor

    if not cpu:
        output = output.cpu()

    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        alpha = input_dict.get("alpha", 1.0)
        training = input_dict.get("training", False)

        if training:
            random_tensor = tf.random.uniform(shape=tf.shape(input_tensor), minval=0, maxval=1)
            mask = tf.cast(random_tensor > p, dtype=tf.float32)
            output = input_tensor * mask + alpha * (1 - mask)
            output = output * (1 / (1 - p))
        else:
            output = input_tensor

        output = output.numpy()

    return {"result": output}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "p": 0.5,
        "alpha": -1.0,
        "training": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()