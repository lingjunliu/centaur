import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from typing import Dict

    class AttributeModule(torch.jit.ScriptModule):
        def __init__(self):
            super().__init__()
            self.foo = torch.jit.Attribute(input_dict["value"], float)
            self.names_ages = torch.jit.Attribute(input_dict["value2"], Dict[str, int])

    class AttributeModule2(torch.jit.ScriptModule):
        def __init__(self):
            super().__init__()
            self.foo = torch.jit.Attribute(input_dict["value"], float)

    class AttributeModule3(torch.jit.ScriptModule):
        def __init__(self):
            super().__init__()
            self.names_ages = torch.jit.Attribute({}, Dict[str, int])

    if input_dict["case"] == 0:
        module = AttributeModule()
        result1 = module.foo
        result2 = module.names_ages
    elif input_dict["case"] == 1:
        module = AttributeModule2()
        result1 = module.foo
        result2 = 0
    else:
        module = AttributeModule3()
        module.names_ages["someone"] = 20
        result1 = 0
        result2 = module.names_ages["someone"]

    return {"result1": np.array(result1), "result2": result2}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    from typing import Dict

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    class AttributeModule:
        def __init__(self):
            self.foo = input_dict["value"]
            self.names_ages = input_dict["value2"]

    class AttributeModule2:
        def __init__(self):
            self.foo = input_dict["value"]

    class AttributeModule3:
        def __init__(self):
            self.names_ages = {}

    with tf.device(device_string):
        if input_dict["case"] == 0:
            module = AttributeModule()
            result1 = module.foo
            result2 = module.names_ages
        elif input_dict["case"] == 1:
            module = AttributeModule2()
            result1 = module.foo
            result2 = 0
        else:
            module = AttributeModule3()
            module.names_ages["someone"] = 20
            result1 = 0
            result2 = module.names_ages["someone"]

    return {"result1": np.array(result1), "result2": result2}

def main():
    A_TOL = 0.01
    input_data = {
        "value": 0.1,
        "value2": {},
        "case": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result1"], tf_result["result1"], atol=A_TOL)
    if isinstance(torch_result["result2"], dict):
        assert len(torch_result["result2"]) == len(tf_result["result2"])
    else:
        assert np.allclose(torch_result["result2"], tf_result["result2"], atol=A_TOL)

    input_data = {
        "value": 0.1,
        "case": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result1"], tf_result["result1"], atol=A_TOL)
    assert np.allclose(torch_result["result2"], tf_result["result2"], atol=A_TOL)

    input_data = {
        "case": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result1"], tf_result["result1"], atol=A_TOL)
    assert np.allclose(torch_result["result2"], tf_result["result2"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()