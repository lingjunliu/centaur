import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import warnings

    message = input_dict["message"]
    category = input_dict.get("category", UserWarning)

    if not cpu:
        pass

    warnings.warn(message, category)

    return {}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import warnings

    message = input_dict["message"]
    category = input_dict.get("category", UserWarning)

    warnings.warn(message, category)

    return {}

def main():
    A_TOL = 0.01
    input_data = {
        "message": "This is a test warning",
        "category": DeprecationWarning
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print("Success")

if __name__ == "__main__":
    main()