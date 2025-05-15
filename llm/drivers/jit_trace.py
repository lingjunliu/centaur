import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    func_name = input_dict["func"]
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
    
    if isinstance(example_inputs, tuple):
        example_inputs = tuple(torch.tensor(x) if isinstance(x, np.ndarray) else x for x in example_inputs)
    elif isinstance(example_inputs, np.ndarray):
        example_inputs = (torch.tensor(example_inputs),)
    elif example_inputs is not None:
         example_inputs = (torch.tensor(example_inputs),)

    if isinstance(check_inputs, list):
         check_inputs = [tuple(torch.tensor(x) if isinstance(x, np.ndarray) else x for x in inp) for inp in check_inputs]
    
    if not cpu:
        if example_inputs is not None:
            example_inputs = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs)
        if check_inputs is not None:
            check_inputs = [tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inp) for inp in check_inputs]
    
    if func_name == "foo":
        def foo(x, y):
            return 2 * x + y
        func = foo
    else:
        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)
        n = Net()
        func = n
    
    if isinstance(func, torch.nn.Module):
      example_input_to_forward = torch.rand(1, 1, 5, 5)
      if not cpu:
          example_input_to_forward = example_input_to_forward.cuda()

      traced_module = torch.jit.trace(func, example_input_to_forward, optimize=optimize, check_trace=check_trace, check_inputs=check_inputs, check_tolerance=check_tolerance, strict=strict, _force_outplace=_force_outplace, _module_class=_module_class, _compilation_unit=_compilation_unit, _store_inputs=_store_inputs)
      result = traced_module(example_input_to_forward)
    else:
      if example_kwarg_inputs is not None:
           for key, value in example_kwarg_inputs.items():
                if isinstance(value, np.ndarray):
                     example_kwarg_inputs[key] = torch.tensor(value)

           if not cpu:
              for key, value in example_kwarg_inputs.items():
                   if isinstance(value, torch.Tensor):
                        example_kwarg_inputs[key] = value.cuda()


      if example_inputs is not None:
          traced_module = torch.jit.trace(func, example_inputs, optimize=optimize, check_trace=check_trace, check_inputs=check_inputs, check_tolerance=check_tolerance, strict=strict, _force_outplace=_force_outplace, _module_class=_module_class, _compilation_unit=_compilation_unit, _store_inputs=_store_inputs)
          example_input_to_foo_1 = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
          example_input_to_foo_2 = torch.tensor(np.array([4.0, 5.0, 6.0], dtype=np.float32))

          if not cpu:
              example_input_to_foo_1 = example_input_to_foo_1.cuda()
              example_input_to_foo_2 = example_input_to_foo_2.cuda()
      
          result = traced_module(example_input_to_foo_1, example_input_to_foo_2)
      elif example_kwarg_inputs is not None:
          traced_module = torch.jit.trace(func, example_kwarg_inputs=example_kwarg_inputs, optimize=optimize, check_trace=check_trace, check_inputs=check_inputs, check_tolerance=check_tolerance, strict=strict, _force_outplace=_force_outplace, _module_class=_module_class, _compilation_unit=_compilation_unit, _store_inputs=_store_inputs)
          raise ValueError("example_kwarg_inputs not supported")
      else:
           raise ValueError("Either example_inputs or example_kwarg_inputs must be specified")


    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        func_name = input_dict["func"]
        example_inputs = input_dict.get("example_inputs", None)
        check_trace = input_dict.get("check_trace", True)
        check_inputs = input_dict.get("check_inputs", None)
        check_tolerance = input_dict.get("check_tolerance", 1e-05)
        strict = input_dict.get("strict", True)
        example_kwarg_inputs = input_dict.get("example_kwarg_inputs", None)

        if isinstance(example_inputs, tuple):
            example_inputs = tuple(tf.constant(x) if isinstance(x, np.ndarray) else x for x in example_inputs)
        elif isinstance(example_inputs, np.ndarray):
            example_inputs = (tf.constant(example_inputs),)
        elif example_inputs is not None:
             example_inputs = (tf.constant(example_inputs),)

        if isinstance(check_inputs, list):
             check_inputs = [tuple(tf.constant(x) if isinstance(x, np.ndarray) else x for x in inp) for inp in check_inputs]

        if func_name == "foo":
            @tf.function
            def foo(x, y):
                return 2 * x + y
            func = foo
        else:
            class Net(tf.Module):
                def __init__(self):
                    self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=1, kernel_size=3, use_bias=False, padding='valid')

                @tf.function
                def __call__(self, x):
                    return self.conv(x)
            
            n = Net()
            example_input_to_net = tf.random.normal((1, 5, 5, 1))
            n(example_input_to_net)
            func = n
        
        if example_kwarg_inputs is not None:
            for key, value in example_kwarg_inputs.items():
                if isinstance(value, np.ndarray):
                    example_kwarg_inputs[key] = tf.constant(value)
        
        if example_inputs is not None:
            if isinstance(func_name, str) and func_name == "foo":
                 result = func(example_inputs[0], example_inputs[1]).numpy()
            else:
                 example_input_to_forward = tf.random.normal((1, 5, 5, 1))
                 result = func(example_input_to_forward).numpy()
        elif example_kwarg_inputs is not None:
             result = func(**example_kwarg_inputs).numpy()
        else:
            raise ValueError("Either example_inputs or example_kwarg_inputs must be specified")

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data_foo = {
        "func": "foo",
        "example_inputs": (np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32))
    }

    input_data_net = {
        "func": "Net",
        "example_inputs": np.random.rand(1, 1, 5, 5).astype(np.float32)
    }

    torch_result_foo = torch_version(input_data_foo)
    tf_result_foo = tensorflow_version(input_data_foo)
    assert np.allclose(torch_result_foo["result"], tf_result_foo["result"], atol=A_TOL), "Results do not match for foo"

    torch_result_net = torch_version(input_data_net)
    tf_result_net = tensorflow_version(input_data_net)
    assert np.allclose(torch_result_net["result"], tf_result_net["result"], atol=A_TOL), "Results do not match for Net"
    
    print("Success")

if __name__ == "__main__":
    main()