import numpy as np
import torch
import torch.nn as nn

def torch_version(input_dict, cpu=True):

    func = input_dict["func"]
    example_inputs = input_dict.get("example_inputs", None)
    optimize = input_dict.get("optimize", None)
    check_trace = input_dict.get("check_trace", True)
    check_inputs = input_dict.get("check_inputs", None)
    check_tolerance = input_dict.get("check_tolerance", 1e-05)
    strict = input_dict.get("strict", True)
    _force_outplace = input_dict.get("_force_outplace", False)
    _module_class = input_dict.get("_module_class", None)
    _compilation_unit = input_dict.get("_compilation_unit", torch.jit.CompilationUnit())
    example_kwarg_inputs = input_dict.get("example_kwarg_inputs", None)
    _store_inputs = input_dict.get("_store_inputs", True)

    if example_inputs is not None:
        example_inputs = tuple(torch.tensor(x) if isinstance(x, np.ndarray) else x for x in example_inputs) if isinstance(example_inputs, tuple) else (torch.tensor(example_inputs),) if isinstance(example_inputs, np.ndarray) else example_inputs
    if check_inputs is not None:
        check_inputs = [tuple(torch.tensor(x) if isinstance(x, np.ndarray) else x for x in example) for example in check_inputs]
    if example_kwarg_inputs is not None:
        example_kwarg_inputs = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in example_kwarg_inputs.items()}
    
    if not cpu:
        if example_inputs is not None:
            example_inputs = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs) if isinstance(example_inputs, tuple) else (example_inputs.cuda(),) if isinstance(example_inputs, torch.Tensor) else example_inputs
        if check_inputs is not None:
            check_inputs = [tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in example) for example in check_inputs]
        if example_kwarg_inputs is not None:
            example_kwarg_inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in example_kwarg_inputs.items()}
    
    traced_func = torch.jit.trace(func, example_inputs=example_inputs, optimize=optimize, check_trace=check_trace, check_inputs=check_inputs, check_tolerance=check_tolerance, strict=strict, _force_outplace=_force_outplace, _module_class=_module_class, _compilation_unit=_compilation_unit, example_kwarg_inputs=example_kwarg_inputs, _store_inputs=_store_inputs)
    
    if not cpu:
        traced_func = traced_func.cpu()
    
    return {"result": traced_func}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    func = input_dict["func"]
    example_inputs = input_dict.get("example_inputs", None)
    optimize = input_dict.get("optimize", None)
    check_trace = input_dict.get("check_trace", True)
    check_inputs = input_dict.get("check_inputs", None)
    check_tolerance = input_dict.get("check_tolerance", 1e-05)
    strict = input_dict.get("strict", True)
    _force_outplace = input_dict.get("_force_outplace", False)
    _module_class = input_dict.get("_module_class", None)
    _compilation_unit = input_dict.get("_compilation_unit", None)
    example_kwarg_inputs = input_dict.get("example_kwarg_inputs", None)
    _store_inputs = input_dict.get("_store_inputs", True)
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        if example_inputs is not None:
            example_inputs = tuple(tf.constant(x) if isinstance(x, np.ndarray) else x for x in example_inputs) if isinstance(example_inputs, tuple) else (tf.constant(example_inputs),) if isinstance(example_inputs, np.ndarray) else example_inputs
        if check_inputs is not None:
            check_inputs = [tuple(tf.constant(x) if isinstance(x, np.ndarray) else x for x in example) for example in check_inputs]
        if example_kwarg_inputs is not None:
            example_kwarg_inputs = {k: tf.constant(v) if isinstance(v, np.ndarray) else v for k, v in example_kwarg_inputs.items()}
    
        def wrapper(*args, **kwargs):
            
            class TFModuleWrapper(nn.Module):
                def __init__(self, tf_module):
                    super().__init__()
                    self.tf_module = tf_module

                def forward(self, x):
                    x = tf.constant(x.numpy())
                    tf_output = self.tf_module(x)
                    return torch.tensor(tf_output.numpy())

            wrapped_module = TFModuleWrapper(func)
            return wrapped_module(*args, **kwargs)
    
    return {"result": wrapper}

def main():
    import tensorflow as tf
    
    A_TOL = 0.01

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 10)

        def forward(self, x):
            return self.linear(x)

    module = MyModule()
    example_input = np.random.rand(1, 5).astype(np.float32)

    input_data = {
        "func": module,
        "example_inputs": (example_input,),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    example_input_torch = torch.tensor(example_input)
    torch_output = torch_result['result'](example_input_torch)
    tf_output = tf_result['result'](example_input_torch)
    
    assert np.allclose(torch_output.detach().numpy(), tf_output.detach().numpy(), atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()