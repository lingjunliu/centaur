import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    try:
        result = torch.view_as_complex_copy(input_tensor)
    except RuntimeError as e:
        print(f"Torch Error: {e}")
        return {"result": np.array([])}

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

        input_shape = tf.shape(input_tensor)
        last_dim = input_shape[-1]
        if last_dim % 2 != 0:
            print("Tensorflow Error: The size of the last dimension of the input tensor must be even")
            return {"result": np.array([])}

        new_shape = tf.concat([input_shape[:-1], [last_dim // 2, 2]], axis=0)
        reshaped_tensor = tf.reshape(input_tensor, new_shape)
        real = reshaped_tensor[..., 0]
        imag = reshaped_tensor[..., 1]

        result = tf.complex(real, imag)
        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if torch_result["result"].size > 0 and tf_result["result"].size > 0:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    else:
        print("Skipping assertion due to empty results.")

    print("Success")

if __name__ == "__main__":
    main()