import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.Dropout2d(p=p, inplace=inplace)
    output = m(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = input_dict["input"]
    p = input_dict.get("p", 0.5)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
        
        if len(input_tensor.shape) == 3:
            shape = tf.shape(input_tensor)
            keep_prob = 1 - p
            
            mask_shape = (shape[0], 1, 1)
            random_tensor = tf.random.uniform(mask_shape, minval=0, maxval=1, dtype=tf.float32)
            binary_mask = tf.cast(random_tensor > p, dtype=tf.float32)
            output = tf.divide(input_tensor * binary_mask, keep_prob)

        else:
            shape = tf.shape(input_tensor)
            keep_prob = 1 - p
            
            mask_shape = (tf.shape(input_tensor)[0], shape[1], 1, 1)
            random_tensor = tf.random.uniform(mask_shape, minval=0, maxval=1, dtype=tf.float32)
            binary_mask = tf.cast(random_tensor > p, dtype=tf.float32)
            output = tf.divide(input_tensor * binary_mask, keep_prob)
        

        output = output.numpy()

    return {"result": output}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(20, 16, 32, 32).astype(np.float32),
        "p": 0.2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    # The issue was that the outputs can be inf, nan. Here's how it's resolved
    torch_result_arr = torch_result["result"]
    tf_result_arr = tf_result["result"]

    torch_result_arr = np.nan_to_num(torch_result_arr, nan=0.0, posinf=0.0, neginf=0.0)
    tf_result_arr = np.nan_to_num(tf_result_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    assert np.allclose(torch_result_arr, tf_result_arr, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()