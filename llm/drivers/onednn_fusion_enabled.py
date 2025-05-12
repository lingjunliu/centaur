import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    result = torch.jit.onednn_fusion_enabled()
    
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        tf.config.optimizer.set_experimental_options({"layout_optimizer": input_dict["val"]})
        result = tf.config.optimizer.get_experimental_options()["layout_optimizer"]
        result = np.array(result)

    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "val": True
    }

    # Torch example
    import torch
    torch.jit.onednn_fusion_enabled()
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "val": False
    }
    
    # Torch example
    torch.jit.onednn_fusion_enabled()
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()