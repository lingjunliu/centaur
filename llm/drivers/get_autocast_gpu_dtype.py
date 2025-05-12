import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    if not cpu and torch.cuda.is_available():
        torch.cuda.init()

    result = torch.get_autocast_gpu_dtype()

    return {"result": str(result).replace("torch.", "")}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        return {"result": str(tf.float16).replace("<class 'tensorflow.python.framework.dtypes.float16'>", "float16")}
    else:
        return {"result": str(tf.bfloat16).replace("<class 'tensorflow.python.framework.dtypes.bfloat16'>", "bfloat16")}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()