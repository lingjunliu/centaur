import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)

    if not cpu:
        input_tensor = input_tensor.cuda()

    dropout = torch.nn.Dropout2d(p=p)
    result = dropout(input_tensor)

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

        shape = tf.shape(input_tensor)
        random_tensor = tf.random.uniform(shape=shape, minval=0, maxval=1.0, dtype=tf.float32)
        keep_mask = random_tensor > p
        output = tf.where(keep_mask, input_tensor / (1 - p), tf.zeros_like(input_tensor, dtype=tf.float32))
        result = output

        result = tf.reshape(result, input_tensor.shape).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5).astype(np.float32),
        "p": 0.3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()