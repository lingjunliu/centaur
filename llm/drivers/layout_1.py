import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = input_tensor.layout

    if not cpu:
        result = str(result)
    else:
        result = str(result)
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])

        # TensorFlow doesn't have a direct equivalent to torch.layout
        # So we'll return a string indicating the layout based on the shape.
        if len(input_tensor.shape) == 0:
            layout = "Scalar"
        elif len(input_tensor.shape) == 1:
            layout = "Vector"
        elif len(input_tensor.shape) == 2:
            layout = "Matrix"
        else:
            layout = "Tensor"
        
        result = layout
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == "torch.strided", "Results do not match"
    assert tf_result["result"] == "Matrix", "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()