import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.FeatureAlphaDropout(p=p)
    m.train(training)
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        training = input_dict.get("training", False)
        alpha = 1.0
        
        if training:
            keep_prob = 1 - p
            input_shape = tf.shape(input_tensor)
            rank = tf.rank(input_tensor)
            
            noise_shape = tf.concat([tf.expand_dims(input_shape[0], axis=0), tf.ones([tf.maximum(rank - 1, 0)], dtype=tf.int32)], axis=0)
            
            random_tensor = keep_prob + tf.random.uniform(noise_shape, 0, 1, dtype=tf.float32)
            binary_tensor = tf.floor(random_tensor)

            output = (input_tensor - alpha) * binary_tensor
            output /= keep_prob
            output += alpha
        else:
            output = input_tensor

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        "p": 0.5,
        "training": True,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        "p": 0.5,
        "training": False,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.float32),
        "p": 0.5,
        "training": False,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()