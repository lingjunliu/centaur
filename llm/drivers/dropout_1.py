import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    train = input_dict.get("train", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.dropout(input_tensor, p=p, training=train)

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
        train = input_dict.get("train", True)

        if train:
            keep_prob = 1 - p
            random_tensor = tf.random.uniform(shape=tf.shape(input_tensor), dtype=tf.float32)
            binary_mask = tf.cast(random_tensor > (1 - keep_prob), dtype=tf.float32)
            output = tf.compat.v1.div(input_tensor, keep_prob) * binary_mask
        else:
            output = input_tensor

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "p": 0.5,
        "train": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data_eval = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "p": 0.5,
        "train": False
    }

    torch_result_eval = torch_version(input_data_eval)
    tf_result_eval = tensorflow_version(input_data_eval)

    assert np.allclose(torch_result_eval["result"], tf_result_eval["result"], atol=A_TOL), "Eval results do not match"

    print("Success")

if __name__ == "__main__":
    main()