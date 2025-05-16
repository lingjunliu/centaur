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
    train = input_dict.get("train", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = F.dropout(input_tensor, p=p, training=train)

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
        input_tensor = tf.constant(input_dict["input"])
        p = input_dict.get("p", 0.5)
        train = input_dict.get("train", True)
        
        if train:
            result = tf.nn.dropout(input_tensor, rate=p)
        else:
            result = tf.identity(input_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056, 0.5], dtype=np.float32),
        "p": 0.5,
        "train": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    # The output values can be different due to the randomness of dropout,
    # so we compare the number of zeroed elements.

    torch_zero_count = np.sum(torch_result["result"] == 0)
    tf_zero_count = np.sum(tf_result["result"] == 0)

    assert abs(torch_zero_count - tf_zero_count) <= 1, "Zero counts do not match"

    print("Success")