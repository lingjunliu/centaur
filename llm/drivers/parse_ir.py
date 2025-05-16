import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = input_dict["input"].astype(str).tolist()
    format_str = input_dict.get("format", "{}{}{}{}")

    try:
        result = torch.parse_ir(format_str.format(*input_tensor))
    except RuntimeError as e:
        result = str(e)

    return {'result': str(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = input_dict["input"].astype(str).tolist()
        format_str = input_dict.get("format", "{}{}{}{}")

        result = format_str.format(*input_tensor)

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.int32),
        "format": "Number {} {} {} {}"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()