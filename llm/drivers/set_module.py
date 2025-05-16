import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    m = torch.jit.script(torch.nn.Linear(5, 5))

    input_module = input_dict.get("input_module", m)
    name = input_dict.get("name", "new_module")

    if not cpu:
        pass

    torch.jit.set_module(input_module, name)

    if not cpu:
        pass
    
    return {"result": input_module}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    class DummyModule(tf.Module):
        def __init__(self):
            super(DummyModule, self).__init__()
            self.dense = tf.keras.layers.Dense(5, kernel_initializer='ones', bias_initializer='zeros')
            self.dense.build(input_shape=(None, 5))

        @tf.function
        def __call__(self, x):
            return self.dense(x)
    
    initial_module = DummyModule()
    name = input_dict.get("name", "new_module")

    setattr(initial_module, name, initial_module)

    return {"result": initial_module}

def main():
    A_TOL = 0.01

    class DummyModule():
        def __init__(self):
            pass

    input_module = DummyModule()
    input_module.lin = "old_module"
    
    input_data = {
        "input_module": input_module,
        "name": "lin"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    print("Success")

if __name__ == "__main__":
    main()