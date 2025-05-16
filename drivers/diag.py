import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    diag_input = torch.tensor(input["input"])
    diagonal = input.get("diagonal", 0)
    
    if cpu:
        diag_input = diag_input.cpu()
        
    result = torch.diag(diag_input, diagonal=diagonal)
    
    return {"diag_output": result.numpy()}

### TensorFlow Implementation

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        diag_input = tf.constant(input["input"])
        diagonal = input.get("diagonal", 0)
        
        if len(diag_input.shape) == 1:
            result = tf.linalg.diag(diag_input)
        else:
            result = tf.linalg.diag_part(tf.roll(diag_input, shift=-diagonal, axis=0))

        return {"diag_output": result.numpy()}

### Main Function Example

def main():
    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "diagonal": 0
    }

    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.array_equal(torch_result["diag_output"], tf_result["diag_output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()