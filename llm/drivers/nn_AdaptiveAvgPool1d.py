import numpy as np
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.AdaptiveAvgPool1d(output_size)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    output_size = input_dict["output_size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = input_tensor.shape
        if len(input_shape) == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        input_length = input_tensor.shape[2]

        def calculate_indices(output_size, input_length):
            indices = []
            step = float(input_length) / output_size
            for i in range(output_size):
                start = int(np.floor(i * step))
                end = int(np.ceil((i + 1) * step))
                indices.append((start, end))
            return indices

        indices = calculate_indices(output_size, input_length)
        pooled_outputs = []

        for start, end in indices:
            slice_tensor = input_tensor[:, :, start:end]
            if tf.size(slice_tensor) > 0:
                mean_val = tf.reduce_mean(slice_tensor, axis=2)
            else:
                mean_val = tf.zeros((input_tensor.shape[0], input_tensor.shape[1]))
            pooled_outputs.append(mean_val)

        result = tf.stack(pooled_outputs, axis=2)
        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 64, 8).astype(np.float32),
        "output_size": 5,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(64, 8).astype(np.float32),
        "output_size": 5,
    }

    torch_result = torch_version(input_data)
    tf_result_data = input_data.copy()
    tf_result_data["input"] = np.expand_dims(tf_result_data["input"], axis=0)
    tf_result = tensorflow_version(tf_result_data)
    tf_result["result"] = np.squeeze(tf_result["result"], axis=0)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()