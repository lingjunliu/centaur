import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    if not cpu and not torch.cuda.is_available():
        return None

    torch.set_warn_always(input_dict.get("value", True))

    return {}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    tf.config.experimental.enable_mlir_bridge()
    tf.config.run_functions_eagerly(True)
    value = input_dict.get("value", True)
    
    if value:
        tf.debugging.enable_check_numerics()
    else:
        tf.debugging.disable_check_numerics()

    return {}


def main():
    A_TOL = 0.01

    input_data = {
        "value": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    print("Success")

if __name__ == "__main__":
    main()