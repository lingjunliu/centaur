import numpy as np
import warnings

def torch_version(input_dict, cpu=True):
    import torch

    message = input_dict["message"]
    category = input_dict.get("category", UserWarning)
    stacklevel = input_dict.get("stacklevel", 2)

    warnings.warn(message, category=category, stacklevel=stacklevel)

    return {"result": np.array([0])}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    message_np = input_dict["message"]
    category = input_dict.get("category", UserWarning)
    stacklevel = input_dict.get("stacklevel", 2)

    message_str = message_np

    warnings.warn(message_str, category=category, stacklevel=stacklevel)

    return {"result": np.array([0])}

def main():
    A_TOL = 0.01

    input_data = {
        "message": "This is a warning message",
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()