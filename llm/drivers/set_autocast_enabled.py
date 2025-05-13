import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    enabled = input_dict["enabled"]

    if not cpu:
        torch.cuda.init()

    torch.set_autocast_enabled(enabled)

    return {"result": torch.is_autocast_enabled()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    enabled = input_dict["enabled"]

    if enabled:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    else:
        tf.keras.mixed_precision.set_global_policy('float32')

    return {"result": tf.keras.mixed_precision.global_policy().name == 'mixed_float16'}

def main():
    A_TOL = 0.01

    input_data = {
        "enabled": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    input_data = {
        "enabled": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()