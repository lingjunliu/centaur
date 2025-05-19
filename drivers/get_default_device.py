import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.set_default_device(device)
    result = torch.get_default_device()

    if not cpu and device == 'cuda':
        result = torch.device('cuda')
    elif not cpu and device == 'cpu':
        result = torch.device('cpu')

    if isinstance(result, torch.device):
        result_str = str(result)
    else:
        result_str = str(result)
        
    if device == 'cuda' and torch.cuda.is_available():
        result_np = 'cuda'
    else:
        result_np = 'cpu'

    return {"result": result_np}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        result = 'cpu'
    else:
        try:
            with tf.device('/GPU:0'):
                _ = tf.constant([0.0])
                result = 'cuda'
        except tf.errors.InvalidArgumentError:
            result = 'cpu'
    
    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {}

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    torch_result_cuda = torch_version(input_data, cpu=False)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    tf_result_cuda = tensorflow_version(input_data, cpu=False)
    
    # Assert to see if they are equal
    assert torch_result["result"] == tf_result["result"], "Results do not match"

    if "cuda" in torch_result_cuda["result"]:
        assert "cuda" in tf_result_cuda["result"], "Results do not match"
    else:
         assert "cpu" in tf_result_cuda["result"], "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()