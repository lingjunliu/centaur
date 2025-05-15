import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    b = torch.tensor(input_dict["b"])
    
    if not cpu:
        A = A.cuda()
        b = b.cuda()
    
    result = torch.linalg.solve(A, b)
    
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
        A = tf.constant(input_dict["A"])
        b = tf.constant(input_dict["b"])
        
        A = tf.cast(A, dtype=tf.float32)
        b = tf.cast(b, dtype=tf.float32)
        
        result = tf.linalg.solve(A, tf.expand_dims(b, axis=-1))
        result = tf.squeeze(result, axis=-1)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "b": np.array([5.0, 11.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()