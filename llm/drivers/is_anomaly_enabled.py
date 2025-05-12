import numpy as np
import logging

def torch_version(input_dict, cpu=True):
    import torch

    if not cpu:
        torch.cuda.init()

    if "enabled" in input_dict:
        enabled = input_dict["enabled"]
    else:
        enabled = True

    torch.set_anomaly_enabled(enabled)

    result = torch.is_anomaly_enabled()
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    tf.config.run_functions_eagerly(True)

    if "enabled" in input_dict:
        enabled = input_dict["enabled"]
    else:
        enabled = True

    if enabled:
        tf.debugging.enable_check_numerics()
        tf.get_logger().setLevel(logging.DEBUG)
    else:
        tf.debugging.disable_check_numerics()
        tf.get_logger().setLevel(logging.INFO)

    level = tf.get_logger().getEffectiveLevel()
    is_enabled = level == logging.DEBUG

    return {"result": np.array(is_enabled)}

def main():
    A_TOL = 0.01

    input_data_true = {
        "enabled": True,
    }

    input_data_false = {
        "enabled": False,
    }

    torch_result_true = torch_version(input_data_true)
    tf_result_true = tensorflow_version(input_data_true)

    assert np.allclose(torch_result_true["result"], tf_result_true["result"], atol=A_TOL), "Results do not match"

    torch_result_false = torch_version(input_data_false)
    tf_result_false = tensorflow_version(input_data_false)

    assert np.allclose(torch_result_false["result"], tf_result_false["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()