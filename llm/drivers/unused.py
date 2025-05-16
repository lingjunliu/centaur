import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    class MyModule(nn.Module):
        def __init__(self, use_memory_efficient):
            super().__init__()
            self.use_memory_efficient = use_memory_efficient

        @torch.jit.unused
        def memory_efficient(self, x):
            return x + 10

        def forward(self, x):
            if self.use_memory_efficient:
                if self.use_memory_efficient:
                    return x + 10 # Fallback behavior, cannot use try/except in torch script.
                else:
                    return x + 10
            else:
                return x + 10

    use_memory_efficient = input_dict['use_memory_efficient']
    input_tensor = torch.tensor(input_dict['input'])

    if not cpu:
        input_tensor = input_tensor.cuda()

    module = MyModule(use_memory_efficient=use_memory_efficient)
    try:
        module = torch.jit.script(module)
        result = module(input_tensor)
        if not cpu:
            result = result.cpu()
        result = result.numpy()
    except RuntimeError as e:
        result = str(e)

    return {'result': result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    class MyModule:
        def __init__(self, use_memory_efficient):
            self.use_memory_efficient = use_memory_efficient

        def memory_efficient(self, x):
            raise RuntimeError("This part of the model is not implemented for tensorflow")

        def forward(self, x):
            if self.use_memory_efficient:
                try:
                    return self.memory_efficient(x)
                except RuntimeError as e:
                    return "RuntimeError: This part of the model is not implemented for tensorflow"
            else:
                return tf.add(x, 10).numpy()

    use_memory_efficient = input_dict['use_memory_efficient']
    input_tensor = tf.constant(input_dict['input'])

    module = MyModule(use_memory_efficient=use_memory_efficient)

    try:
        result = module.forward(input_tensor)
        if not isinstance(result, str):
            pass
    except RuntimeError as e:
        result = str(e)

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data_true = {
        'use_memory_efficient': True,
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    input_data_false = {
        'use_memory_efficient': False,
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    torch_result_true = torch_version(input_data_true)
    tf_result_true = tensorflow_version(input_data_true)

    torch_result_false = torch_version(input_data_false)
    tf_result_false = tensorflow_version(input_data_false)
    
    assert isinstance(torch_result_true['result'], np.ndarray), "Results should be numpy arrays."
    assert isinstance(tf_result_true['result'], str), "Results should be strings."
    assert "RuntimeError" in str(tf_result_true['result']), "Results should contain RuntimeError"
    
    assert np.allclose(torch_result_false['result'], tf_result_false['result'], atol=A_TOL), "Results do not match."

    print("Success")

if __name__ == "__main__":
    main()