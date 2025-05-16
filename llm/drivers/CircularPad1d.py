import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    input_tensor = input_tensor.unsqueeze(0)
    circular_pad = torch.nn.CircularPad1d(padding)
    result = circular_pad(input_tensor)
    result = result.squeeze(0)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]
    input_shape = input_tensor.shape.as_list()

    if len(input_shape) == 0:
      input_tensor = tf.reshape(input_tensor, [1])
      input_shape = input_tensor.shape.as_list()


    def circular_pad_tf(tensor, padding_size):
        length = tf.shape(tensor)[0]
        
        prefix = tensor[-padding_size:]
        suffix = tensor[:padding_size]

        padded_tensor = tf.concat([prefix, tensor, suffix], axis=0)
        return padded_tensor
    
    padded_tensor = circular_pad_tf(input_tensor, padding)
    

    return {"result": padded_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "padding": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()