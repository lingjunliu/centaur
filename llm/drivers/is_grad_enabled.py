import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    torch.set_grad_enabled(input_dict.get("mode", True))
    result = torch.is_grad_enabled()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    mode = input_dict.get("mode", True)
    
    if mode:
        with tf.GradientTape() as tape:
            result = True
    else:
        result = False

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "mode": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()