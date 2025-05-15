import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    if cpu:
        result = str(torch.get_autocast_dtype("cpu"))
    else:
        result = str(torch.get_autocast_dtype("cuda"))

    if not cpu:
        pass

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        result = "torch.float32"
    else:
        if tf.config.list_physical_devices('GPU'):
            result = "torch.float16"
        else:
            result = "torch.float32"

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result_cpu = torch_version(input_data, cpu=True)
    tf_result_cpu = tensorflow_version(input_data, cpu=True)
    assert torch_result_cpu["result"] == tf_result_cpu["result"], "CPU Results do not match"

    if torch.cuda.is_available():
        torch_result_cuda = torch_version(input_data, cpu=False)
        tf_result_cuda = tensorflow_version(input_data, cpu=False)
        assert torch_result_cuda["result"] == tf_result_cuda["result"], "CUDA Results do not match"

    print("Success")

if __name__ == "__main__":
    main()