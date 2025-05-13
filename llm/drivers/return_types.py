import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    # Unpack inputs from dictionary
    # Assume the input dictionary contains the parameter values for the api (e.g. input_tensor = input['input'])
    # Assume the input dictionary contains values in numpy format and convert them to torch format (e.g. input_tensor = torch.tensor(input['input]))
    # For optional arguments, use the default value explicitly mentioned in the documentation in case the dictionary does not contain the argument's value (e.g. alpha = input_dict.get("alpha", 1.0))
    # Do not apply any additional correction on the arguments (e.g. Do not force "index" arguments to be IntTensor, leave them as tensors)
    # If cpu is set to false (which means it needs to be run on cuda), convert cpu objects to cuda object (e.g. if not cpu: input_tensor = input_tensor.cuda())
    # The input dict will never have "out" or "device", so do not expect these to be in the dict. If the API takes both or one of these as parameters, ignore them
    # Run the api and get the output
    # If cpu is set to false, convert the output to its cpu version (e.g. if not cpu: result = result.cpu())
    # Return a dictionary containing the results in numpy version (e.g. return { 'result': result.numpy() })
    
    
    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.fft(input_tensor)
    
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
        input_tensor = tf.constant(input_dict["input"])

        result = tf.signal.fft(tf.cast(input_tensor, tf.complex64))
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()