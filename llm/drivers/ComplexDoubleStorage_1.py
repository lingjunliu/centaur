import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    source = torch.tensor(input_dict["source"], dtype=torch.float64)
    
    if not cpu:
        source = source.cuda()
    
    source_np = source.numpy()
    complex_data = np.zeros(source_np.shape[0] // 2, dtype=np.complex128)
    complex_data.real = source_np[::2]
    complex_data.imag = source_np[1::2]
    
    if not cpu:
        pass
    
    return {"result": complex_data}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    source = tf.constant(input_dict["source"], dtype=tf.float64)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        
        source_np = source.numpy()
        
        complex_array = np.zeros(source_np.shape[0] // 2, dtype=np.complex128)
        complex_array.real = source_np[::2]
        complex_array.imag = source_np[1::2]
    
    return {"result": complex_array}

def main():
    A_TOL = 0.01
    
    input_data = {
        "source": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()