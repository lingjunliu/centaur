import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    modules = input_dict.get("modules", None)

    if modules is not None:
        if isinstance(modules, dict):
            torch_modules = {k: nn.Conv2d(1, 1, 1) if isinstance(v, str) and v == "conv" else nn.MaxPool2d(3) for k, v in modules.items()}
        else:
            torch_modules = [('lrelu', nn.LeakyReLU()) if isinstance(item[1], str) and item[1] == "lrelu" else ('prelu', nn.PReLU()) for item in modules]
    else:
        torch_modules = None

    module_dict = nn.ModuleDict(torch_modules)

    if not cpu:
        for key, module in module_dict.items():
            module_dict[key] = module.cuda()
    
    #clear
    if "clear" in input_dict and input_dict["clear"]:
        module_dict.clear()

    #pop
    if "pop_key" in input_dict:
        popped_module = module_dict.pop(input_dict["pop_key"])
        if not cpu:
            popped_module = popped_module.cpu()

    #update
    if "update_modules" in input_dict:
        update_modules = input_dict["update_modules"]
        if isinstance(update_modules, dict):
            torch_update_modules = {k: nn.Conv2d(1, 1, 1) if isinstance(v, str) and v == "conv" else nn.MaxPool2d(3) for k, v in update_modules.items()}
        else:
            torch_update_modules = [('lrelu', nn.LeakyReLU()) if isinstance(item[1], str) and item[1] == "lrelu" else ('prelu', nn.PReLU()) for item in update_modules]
        
        module_dict.update(torch_update_modules)

    if not cpu:
        result_dict = {}
        for key, module in module_dict.items():
            module_dict[key] = module.cpu()

    items = list(module_dict.items())
    keys = list(module_dict.keys())
    values = list(module_dict.values())
    
    if not cpu:
        items_cpu = []
        for k, v in items:
          items_cpu.append((k, v.cpu()))
        
        values_cpu = []
        for v in values:
          values_cpu.append(v.cpu())

        return {
            "items": [(k, str(v)) for k, v in items_cpu],
            "keys": keys,
            "values": [str(v) for v in values_cpu]
        }
    else:
        return {
            "items": [(k, str(v)) for k, v in items],
            "keys": keys,
            "values": [str(v) for v in values]
        }


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    class TFModuleDict:
        def __init__(self, modules=None):
            self._modules = {}
            if modules is not None:
                if isinstance(modules, dict):
                    self._modules.update(modules)
                else:
                    for k, v in modules:
                        self._modules[k] = v

        def __getitem__(self, key):
            return self._modules[key]

        def __setitem__(self, key, module):
            self._modules[key] = module

        def __delitem__(self, key):
            del self._modules[key]
        
        def clear(self):
            self._modules.clear()
        
        def pop(self, key):
            return self._modules.pop(key)

        def update(self, modules):
            if isinstance(modules, dict):
                self._modules.update(modules)
            else:
                for k, v in modules:
                    self._modules[k] = v

        def items(self):
            return self._modules.items()
        
        def keys(self):
            return self._modules.keys()
        
        def values(self):
            return self._modules.values()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        modules = input_dict.get("modules", None)

        if modules is not None:
            if isinstance(modules, dict):
                tf_modules = {k: tf.keras.layers.Conv2D(1, (1, 1)) if isinstance(v, str) and v == "conv" else tf.keras.layers.MaxPool2D(3) for k, v in modules.items()}
            else:
                tf_modules = [('lrelu', tf.keras.layers.LeakyReLU()) if isinstance(item[1], str) and item[1] == "lrelu" else ('prelu', tf.keras.layers.PReLU()) for item in modules]
        else:
            tf_modules = None

        module_dict = TFModuleDict(tf_modules)
        
        #clear
        if "clear" in input_dict and input_dict["clear"]:
            module_dict.clear()
        
        #pop
        if "pop_key" in input_dict:
            module_dict.pop(input_dict["pop_key"])

        #update
        if "update_modules" in input_dict:
            update_modules = input_dict["update_modules"]
            if isinstance(update_modules, dict):
                tf_update_modules = {k: tf.keras.layers.Conv2D(1, (1, 1)) if isinstance(v, str) and v == "conv" else tf.keras.layers.MaxPool2D(3) for k, v in update_modules.items()}
            else:
                tf_update_modules = [('lrelu', tf.keras.layers.LeakyReLU()) if isinstance(item[1], str) and item[1] == "lrelu" else ('prelu', tf.keras.layers.PReLU()) for item in update_modules]
            
            module_dict.update(tf_update_modules)

        items = list(module_dict.items())
        keys = list(module_dict.keys())
        values = list(module_dict.values())
        
        return {
            "items": [(k, str(v)) for k, v in items],
            "keys": keys,
            "values": [str(v) for v in values]
        }


def main():
    A_TOL = 0.01
    
    input_data = {
        "modules": {
            "conv1": "conv",
            "pool1": "pool"
        },
        "clear": False,
        "pop_key": "pool1",
        "update_modules": {
            "relu1": "lrelu",
            "prelu1": "prelu"
        }
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)

    assert torch_result["keys"] == tf_result["keys"]

    print("Success")

if __name__ == "__main__":
    main()