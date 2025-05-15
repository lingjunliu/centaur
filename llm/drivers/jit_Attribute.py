import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from typing import Dict

    class AttributeModule(torch.jit.ScriptModule):
        def __init__(self) -> None:
            super().__init__()
            self.foo = torch.jit.Attribute(input_dict["foo_value"], float)
            self.names_ages = torch.jit.Attribute(input_dict["names_ages_value"], Dict[str, int])
            self.names_ages["someone"] = input_dict["someone_age"]

    module = AttributeModule()

    if not cpu:
        pass

    foo = module.foo
    names_ages = module.names_ages

    if not cpu:
        pass

    return {"foo": foo, "names_ages": names_ages}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    from typing import Dict

    class AttributeModule:
        def __init__(self) -> None:
            self.foo = input_dict["foo_value"]
            self.names_ages = input_dict["names_ages_value"].copy()
            self.names_ages["someone"] = input_dict["someone_age"]

    module = AttributeModule()

    foo = module.foo
    names_ages = module.names_ages

    return {"foo": foo, "names_ages": names_ages}


def main():
    A_TOL = 0.01

    input_data = {
        "foo_value": 0.1,
        "names_ages_value": {},
        "someone_age": 20
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert abs(torch_result["foo"] - tf_result["foo"]) < A_TOL
    assert torch_result["names_ages"] == tf_result["names_ages"]

    print("Success")


if __name__ == "__main__":
    main()