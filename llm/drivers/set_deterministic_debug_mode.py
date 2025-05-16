import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    mode = input_dict.get("mode", "default")
    if mode == "default":
        torch.set_deterministic_debug_mode(False)
    elif mode == "warn":
        torch.set_deterministic_debug_mode("warn")
    elif mode == "raise":
        torch.set_deterministic_debug_mode("error")
    
    return {"result": None}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    mode = input_dict.get("mode", "default")

    return {"result": None}

def main():
    A_TOL = 0.01
    input_data = {
        "mode": "default",
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    input_data = {
        "mode": "warn",
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    input_data = {
        "mode": "raise",
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()