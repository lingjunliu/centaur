import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    fill_value = input_dict.get("fill_value", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    padding = input_dict["padding"]
    if len(input_tensor.shape) == 2:
        result = torch.nn.functional.pad(input_tensor, (padding[0], padding[1], padding[2], padding[3]), mode=input_dict.get("mode", "constant"), value=fill_value)
    elif len(input_tensor.shape) == 3:
        result = torch.nn.functional.pad(input_tensor, (padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]), mode=input_dict.get("mode", "constant"), value=fill_value)
    else:
        raise ValueError("Unsupported tensor shape")

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        padding = input_dict["padding"]
        if len(input_tensor.shape) == 2:
            np_padding = np.array([[0, 0], [padding[0], padding[1]], [padding[2], padding[3]]])
        elif len(input_tensor.shape) == 3:
            np_padding = np.array([[0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [padding[4], padding[5]]])
        else:
            raise ValueError("Unsupported tensor shape")

        padding = tf.constant(np_padding)
        mode = input_dict.get("mode", "CONSTANT")
        fill_value = input_dict.get("fill_value", 0)

        if mode == "constant":
            mode = "CONSTANT"
        elif mode == "reflect":
            mode = "REFLECT"
        elif mode == "replicate":
            mode = "SYMMETRIC"
        
        result = tf.pad(input_tensor, padding, mode=mode, constant_values=fill_value)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "padding": [1, 1, 1, 1],
        "mode": "constant",
        "fill_value": 0
    }

    torch_result = torch_version({"input":input_data['input'], "padding":[1,1,1,1], "mode":"constant", "fill_value":0})
    tf_result = tensorflow_version({"input":input_data['input'], "padding":input_data['padding'], "mode":"constant", "fill_value":0})

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "padding": [1, 1, 1, 1],
        "mode": "reflect",
    }

    torch_result = torch_version({"input":input_data['input'], "padding":[1,1,1,1], "mode":"reflect"})
    tf_result = tensorflow_version({"input":input_data['input'], "padding":input_data['padding'], "mode":"reflect"})

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "padding": [1, 1, 1, 1],
        "mode": "replicate",
    }

    torch_result = torch_version({"input":input_data['input'], "padding":[1,1,1,1], "mode":"replicate"})
    tf_result = tensorflow_version({"input":input_data['input'], "padding":input_data['padding'], "mode":"replicate"})

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()