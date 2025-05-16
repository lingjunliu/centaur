import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    class MyModule(torch.jit.ScriptModule):
        def __init__(self):
            super().__init__()

        @torch.jit.script_method
        def forward(self, x):
            return x + 1

    module = MyModule()
    
    if not cpu:
        module = module.cuda()
    
    
    x = torch.tensor(input_dict['x']).to(device)

    result = module.forward(x)
    
    if not cpu:
        result = result.cpu()
    
    return { 'result': result.numpy() }

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
        
    with tf.device(device_string):
        x = tf.constant(input_dict['x'])

        result = tf.add(x, 1)
        
        result = result.numpy()
    
    return { 'result': result }

def main():
    A_TOL = 0.01

    input_data = {
        'x': np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()