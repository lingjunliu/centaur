import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    set_grad_enabled = input_dict["set_grad_enabled"]

    torch.set_grad_enabled(set_grad_enabled)

    is_grad_enabled = torch.is_grad_enabled()

    return {"result": np.array(is_grad_enabled)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    set_grad_enabled = input_dict["set_grad_enabled"]
    
    # TensorFlow doesn't have a direct equivalent, so we'll use a tf.Variable
    # and control its trainable property. This isn't exactly the same, but it's
    # the closest analog for enabling/disabling gradients.

    tf.config.run_functions_eagerly(not set_grad_enabled)

    if set_grad_enabled:
        is_grad_enabled = True
    else:
        is_grad_enabled = False

    return {"result": np.array(is_grad_enabled)}

def main():
    A_TOL = 0.01

    input_data = {
        "set_grad_enabled": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "set_grad_enabled": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()