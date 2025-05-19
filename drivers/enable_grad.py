import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    x = torch.tensor(input_dict["x"], requires_grad=True)

    if not cpu:
        x = x.cuda()

    with torch.no_grad():
        with torch.enable_grad():
            y = x * 2

    if not cpu:
        y = y.cpu()

    return {"result": y.requires_grad}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    x = tf.Variable(input_dict["x"], trainable=True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        with tf.GradientTape() as tape:
            y = x * 2

    return {"result": tape.gradient(y, x) is not None}

def main():
    A_TOL = 0.01

    input_data = {
        "x": np.array([1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()