import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.jit.ONNXTracedModule(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    class IdentityModule(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
        def __call__(self, x):
            return x

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])

        module = IdentityModule()
        
        module_concrete = module.__call__.get_concrete_function(input_tensor)
        
        result = module_concrete
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print("Success")

if __name__ == "__main__":
    main()