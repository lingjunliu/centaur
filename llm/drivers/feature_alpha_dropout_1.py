import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.feature_alpha_dropout(input_tensor, p=p, training=training, inplace=inplace)

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
        alpha = input_dict.get("alpha", 1.0)

        if training:
            keep_prob = 1 - p
            random_tensor = tf.random.uniform(shape=tf.shape(input_tensor), minval=0, maxval=1.0, dtype=tf.float32)
            binary_mask = tf.floor(random_tensor + keep_prob)
            output = (input_tensor * binary_mask) / keep_prob

            alpha_drop = -alpha * tf.ones_like(input_tensor)
            output = tf.where(tf.equal(binary_mask, 0.0), alpha_drop, output)
        else:
            output = input_tensor

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "p": 0.5,
        "training": True,
        "alpha": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "p": 0.5,
        "training": False,
        "alpha": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()