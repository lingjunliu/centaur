import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    dropout = torch.nn.Dropout1d(p=p)
    dropout.train(training)
    result = dropout(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        input_tensor = tf.reshape(input_tensor, (1, -1, 1))
        p = input_dict.get("p", 0.5)
        training = input_dict.get("training", True)

        if training:
            result = tf.nn.dropout(input_tensor, rate=p, noise_shape=[tf.shape(input_tensor)[0], tf.shape(input_tensor)[1], 1])
        else:
            result = input_tensor

        result = tf.squeeze(result).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "p": 0.5,
        "training": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()