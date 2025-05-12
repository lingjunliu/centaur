import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input = torch.tensor(input_dict["input"])

    if not cpu:
        input = input.cuda()

    result = torch.jit.is_scripting()

    if not cpu:
        result = torch.tensor(result).cpu().numpy()
    else:
        result = np.array(result)

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input = input_dict["input"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        result = False

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()