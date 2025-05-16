import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.functional as F

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = F.alpha_dropout(input_tensor, p=p, training=training)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        training = input_dict.get("training", False)

        if training:
            keep_prob = 1 - p
            alpha = -np.sqrt((1 - keep_prob) / keep_prob)
            rand = tf.random.normal(tf.shape(input_tensor))
            mask = tf.cast(tf.greater(rand, alpha), dtype=tf.float32)
            dropped = input_tensor * mask
            mean = 0
            variance = p * np.power(alpha - mean, 2) + keep_prob * np.power(0 - mean, 2)
            b = np.sqrt((1-p) / variance)

            output = b * tf.where(tf.equal(mask,1.0), dropped, alpha * tf.ones_like(input_tensor))
        else:
            output = input_tensor

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

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