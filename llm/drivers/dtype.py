import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = input_tensor.dtype

    if not cpu:
        result = result
    
    return {"result": str(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])

        result = input_tensor.dtype

        result = str(result).split(" ")[1].split("'")[1]
        if result == 'float32':
            result = "torch.float32"
        elif result == 'float64':
             result = "torch.float64"
        elif result == 'int32':
            result = "torch.int32"
        elif result == 'int64':
             result = "torch.int64"
        elif result == 'bool':
            result = "torch.bool"
        elif result == 'int8':
            result = "torch.int8"
        elif result == 'uint8':
            result = "torch.uint8"
        elif result == 'float16':
             result = "torch.float16"
        elif result == 'bfloat16':
            result = "torch.bfloat16"
        elif result == 'complex64':
            result = "torch.complex64"
        elif result == 'complex128':
            result = "torch.complex128"
        elif result == 'int16':
            result = "torch.int16"

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()