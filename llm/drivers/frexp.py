import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    mantissa, exponent = torch.frexp(input_tensor)
    
    if not cpu:
        mantissa = mantissa.cpu()
        exponent = exponent.cpu()
    
    return {"mantissa": mantissa.numpy(), "exponent": exponent.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        
        abs_input = tf.abs(input_tensor)
        
        # Handle zero case
        is_zero = tf.equal(abs_input, 0.0)
        
        # Calculate exponent for non-zero values
        safe_input = tf.where(is_zero, tf.ones_like(abs_input), abs_input)
        exponent = tf.math.floor(tf.math.log(safe_input) / tf.math.log(2.0))
        exponent = tf.where(is_zero, tf.zeros_like(exponent), exponent)  # Set exponent to 0 for zero inputs
        
        mantissa = input_tensor / (tf.pow(2.0, exponent))
        
        # Adjust mantissa and exponent to ensure mantissa is in range [0.5, 1) or [-1, -0.5) for non-zero values
        adjust_mask = tf.logical_and(tf.greater_equal(tf.abs(mantissa), 1.0), tf.logical_not(is_zero))
        
        mantissa_list = tf.unstack(mantissa)
        exponent_list = tf.unstack(exponent)
        
        for i in range(len(mantissa_list)):
            if adjust_mask[i]:
                mantissa_list[i], exponent_list[i] = mantissa_list[i] / 2.0, exponent_list[i] + 1.0
        
        mantissa = tf.stack(mantissa_list)
        exponent = tf.stack(exponent_list)
        
        mantissa = mantissa.numpy()
        exponent = exponent.numpy()
        
        exponent = exponent.astype(np.int32) # Match torch's exponent type

        return {"mantissa": mantissa, "exponent": exponent}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["mantissa"], tf_result["mantissa"], atol=A_TOL), "Mantissa results do not match"
    assert np.allclose(torch_result["exponent"], tf_result["exponent"], atol=A_TOL), "Exponent results do not match"

    print("Success")

if __name__ == "__main__":
    main()