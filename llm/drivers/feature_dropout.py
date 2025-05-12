import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    train = input_dict.get("train", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if train:
        shape = input_tensor.shape[:-1] + (1,)
        mask = (torch.rand(shape, device=input_tensor.device) > p).float()
        mask = mask.expand_as(input_tensor)
        result = input_tensor * mask / (1 - p)
    else:
        result = input_tensor

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        train = input_dict.get("train", False)

        if train:
            shape = tf.shape(input_tensor)
            mask_shape = tf.shape(input_tensor)[:-1]
            random_tensor = tf.random.uniform(shape=mask_shape, dtype=tf.float32)
            keep_prob = 1 - p
            binary_tensor = tf.floor(random_tensor + keep_prob)
            binary_tensor = tf.expand_dims(binary_tensor, axis=-1)
            binary_tensor = tf.tile(binary_tensor, [1 for _ in range(len(shape) - 1)] + [shape[-1]])
            output = tf.divide(input_tensor, keep_prob) * tf.cast(binary_tensor, dtype=tf.float32)
            result = output
        else:
            result = tf.identity(input_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32),
        "p": 0.5,
        "train": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()