import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    result = torch.fmax(input_tensor, other_tensor)

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
        other_tensor = tf.constant(input_dict["other"])

        input_is_nan = tf.math.is_nan(input_tensor)
        other_is_nan = tf.math.is_nan(other_tensor)

        both_nan = tf.logical_and(input_is_nan, other_is_nan)
        only_input_nan = tf.logical_and(input_is_nan, tf.logical_not(other_is_nan))
        only_other_nan = tf.logical_and(tf.logical_not(input_is_nan), other_is_nan)

        result = tf.where(both_nan, tf.constant(np.nan, dtype=input_tensor.dtype),
                           tf.where(only_input_nan, other_tensor,
                                    tf.where(only_other_nan, input_tensor,
                                             tf.maximum(input_tensor, other_tensor))))
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([9.7, float('nan'), 3.1, float('nan')], dtype=np.float32),
        "other": np.array([-2.2, 0.5, float('nan'), float('nan')], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()