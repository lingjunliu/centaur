import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    attn_mask = torch.tensor(input_dict["attn_mask"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        attn_mask = attn_mask.cuda()
    
    union = torch.where(attn_mask == 0, torch.tensor(float('-inf')), input_tensor)
    
    if not cpu:
        union = union.cpu()
    
    return {"result": union.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        attn_mask = tf.cast(tf.constant(input_dict["attn_mask"]), dtype=tf.float32)
        
        result = tf.where(attn_mask == 0, float('-inf'), input_tensor)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "attn_mask": np.array([1, 0, 1, 0], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()