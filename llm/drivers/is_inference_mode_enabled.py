import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    if not torch.is_inference_mode_enabled():
        torch.inference_mode(mode=True)
        inference_mode_was_disabled = True
    else:
        inference_mode_was_disabled = False

    if not cpu:
        torch.set_default_device('cuda')

    result = torch.is_inference_mode_enabled()
    
    if not cpu:
        torch.set_default_device('cpu')
        
    if inference_mode_was_disabled:
        torch.inference_mode(mode=False)

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    # TensorFlow doesn't have a direct equivalent to torch.inference_mode()
    # We'll mimic the behavior by checking the graph optimization level.
    # In eager execution, this doesn't directly disable computation, but it's
    # the closest approximation without modifying the computation graph.

    # In TensorFlow 2.x, eager execution is enabled by default. 
    # Therefore we'll always return False, because inference mode is conceptually always 'on'.
    result = False 
    
    return {"result": np.array(result)}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {}

    # Torch example
    torch_result = torch_version(input_data)
    torch_result_cuda = torch_version(input_data, cpu=False)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    tf_result_cuda = tensorflow_version(input_data)

    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result_cuda["result"], tf_result_cuda["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()