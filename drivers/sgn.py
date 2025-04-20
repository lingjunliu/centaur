import numpy as np

def torch_sgn(input, cpu=True):
    import torch

    # Convert input to a PyTorch tensor
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply the torch.sgn function
    result_tensor = torch.sgn(input_tensor)
    
    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"sgn_result": result_tensor.numpy()}

def tensorflow_sgn(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert input to a TensorFlow tensor
        input_tensor = tf.constant(input["input"], dtype=tf.complex64 if np.iscomplexobj(input["input"]) else tf.float32)

        if not np.iscomplexobj(input["input"]):
            # For non-complex tensors, simply use tf.math.sign
            result_tensor = tf.math.sign(input_tensor)
        else:
            # For complex tensors, normalize to unit magnitude
            angles = tf.math.angle(input_tensor)
            magnitudes = tf.where(tf.equal(tf.abs(input_tensor), 0), tf.zeros_like(tf.abs(input_tensor)), tf.ones_like(tf.abs(input_tensor)))
            real_part = magnitudes * tf.math.cos(angles)
            imag_part = magnitudes * tf.math.sin(angles)
            result_tensor = tf.complex(real_part, imag_part)

        return {"sgn_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([3+4j, 7-24j, 0, 1+2j], dtype=np.complex64)
    }

    # Torch example
    torch_result = torch_sgn(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_sgn(input_data)
    print("TensorFlow result:", tf_result)

    if np.allclose(torch_result["sgn_result"], tf_result["sgn_result"], atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()