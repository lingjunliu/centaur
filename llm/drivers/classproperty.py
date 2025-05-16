import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not hasattr(torch, 'classproperty'):
        class classproperty:
            def __init__(self, f):
                self.f = f
            def __get__(self, obj, owner):
                return self.f(owner)
        torch.classproperty = classproperty

    class Foo:
        @torch.classproperty
        def foo(cls):
            return cls.__name__
    
    if not cpu:
        torch.set_default_device('cuda')

    result = Foo.foo
    if not cpu:
        torch.set_default_device('cpu')
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    class classproperty:
        def __init__(self, f):
            self.f = f
        def __get__(self, obj, owner):
            return self.f(owner)

    class Foo:
        @classproperty
        def foo(cls):
            return cls.__name__
    
    result = Foo.foo
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()