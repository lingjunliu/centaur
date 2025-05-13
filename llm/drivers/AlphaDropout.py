import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.AlphaDropout(p=p)
    if input_dict.get("training", False):
      m.train()
    else:
      m.eval()
    result = m(input_tensor)

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
        training = input_dict.get("training", False)

        if training:
            keep_prob = 1 - p
            alpha = -tf.sqrt((1 - keep_prob) / keep_prob)

            random_tensor = tf.random.uniform(shape=tf.shape(input_tensor), dtype=tf.float32)
            binary_tensor = tf.cast(random_tensor >= (1 - p), dtype=tf.float32)

            mean = tf.reduce_mean(input_tensor)
            variance = tf.reduce_mean(tf.square(input_tensor - mean))
            ret = (input_tensor - mean) / tf.sqrt(variance + 1e-10)
            output = (ret * binary_tensor) + (alpha * (1 - binary_tensor))
        else:
            output = input_tensor

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.1

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "p": 0.5,
        "training": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()