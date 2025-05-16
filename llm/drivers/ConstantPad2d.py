import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.pad(input_tensor, padding, mode='constant', value=value)

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
        padding = input_dict["padding"]
        value = input_dict.get("value", 0.0)

        rank = len(input_tensor.shape)
        paddings = []
        if rank == 2:
            paddings = [[padding[2], padding[3]], [padding[0], padding[1]]]
        elif rank == 3:
            paddings = [[padding[4], padding[5]], [padding[2], padding[3]], [padding[0], padding[1]]]
        elif rank == 4:
            paddings = [[padding[6], padding[7]], [padding[4], padding[5]], [padding[2], padding[3]], [padding[0], padding[1]]]
        elif rank == 5:
            paddings = [[padding[8], padding[9]], [padding[6], padding[7]], [padding[4], padding[5]], [padding[2], padding[3]], [padding[0], padding[1]]]

        result = tf.pad(input_tensor, paddings, constant_values=value)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "padding": (1, 1, 2, 0),
        "value": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()