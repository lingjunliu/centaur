import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
from collections import OrderedDict

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    parameters_dict = {}
    for key, value in input_dict["parameters"].items():
        parameters_dict[key] = nn.Parameter(torch.tensor(value))

    params = nn.ParameterDict(parameters_dict)
    
    if not cpu:
        for key in params:
            params[key] = torch.nn.Parameter(params[key].cuda())

    if "clear" in input_dict and input_dict["clear"]:
        params.clear()
        result = len(params)
    elif "copy" in input_dict:
        copied_params = params.copy()
        result = {k: v.detach().cpu().numpy() for k, v in copied_params.items()}
    elif "fromkeys" in input_dict:
        keys = input_dict["fromkeys"]["keys"]
        default_value = input_dict["fromkeys"].get("default", None)
        if default_value is not None:
             default = nn.Parameter(torch.tensor(default_value))
        else:
            default = None
        
        new_params = nn.ParameterDict({k: default for k in keys})
        result = {k: v.detach().cpu().numpy() if v is not None else None for k, v in new_params.items()}
    elif "get" in input_dict:
        key = input_dict["get"]["key"]
        default_value = input_dict["get"].get("default", None)
        if default_value is not None:
            default = nn.Parameter(torch.tensor(default_value))
        else:
            default = None
        
        value = params.get(key, default)
        result = value.detach().cpu().numpy() if value is not None and isinstance(value, torch.nn.Parameter) else value
    elif "items" in input_dict:
        items = list(params.items())
        result = {k: v.detach().cpu().numpy() for k, v in items}
    elif "keys" in input_dict:
        keys = list(params.keys())
        result = keys
    elif "pop" in input_dict:
        key = input_dict["pop"]["key"]
        value = params.pop(key)
        result = value.detach().cpu().numpy()
    elif "popitem" in input_dict:
        try:
            key, value = params.popitem()
            result = (key, value.detach().cpu().numpy())
        except KeyError:
            result = "KeyError"
    elif "setdefault" in input_dict:
        key = input_dict["setdefault"]["key"]
        default_value = input_dict["setdefault"].get("default", None)
        if default_value is not None:
            default = nn.Parameter(torch.tensor(default_value))
        else:
            default = None
            
        value = params.setdefault(key, default)
        result = value.detach().cpu().numpy() if value is not None and isinstance(value, torch.nn.Parameter) else value
    elif "update" in input_dict:
        update_dict = {}
        for key, value in input_dict["update"]["parameters"].items():
            update_dict[key] = nn.Parameter(torch.tensor(value))
        params.update(update_dict)
        result = {k: v.detach().cpu().numpy() for k, v in params.items()}
    elif "values" in input_dict:
        values = list(params.values())
        result = [v.detach().cpu().numpy() for v in values]
    else:
        result = {k: v.detach().cpu().numpy() for k, v in params.items()}

    if not cpu and isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu().numpy()

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from collections import OrderedDict

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        parameters_dict = {}
        for key, value in input_dict["parameters"].items():
            parameters_dict[key] = tf.Variable(value)

        
        if "clear" in input_dict and input_dict["clear"]:
            parameters_dict.clear()
            result = len(parameters_dict)
        elif "copy" in input_dict:
            copied_params = parameters_dict.copy()
            result = {k: v.numpy() for k, v in copied_params.items()}
        elif "fromkeys" in input_dict:
            keys = input_dict["fromkeys"]["keys"]
            default_value = input_dict["fromkeys"].get("default", None)
            new_params = {}
            for key in keys:
                if default_value is not None:
                    new_params[key] = tf.Variable(default_value)
                else:
                    new_params[key] = None
            result = {k: v.numpy() if v is not None else None for k, v in new_params.items()}
        elif "get" in input_dict:
            key = input_dict["get"]["key"]
            default_value = input_dict["get"].get("default", None)

            if key in parameters_dict:
                result = parameters_dict[key].numpy()
            elif default_value is not None:
                result = default_value
            else:
                result = None

        elif "items" in input_dict:
            items = list(parameters_dict.items())
            result = {k: v.numpy() for k, v in items}
        elif "keys" in input_dict:
            keys = list(parameters_dict.keys())
            result = keys
        elif "pop" in input_dict:
            key = input_dict["pop"]["key"]
            value = parameters_dict.pop(key)
            result = value.numpy()
        elif "popitem" in input_dict:
            try:
                key, value = parameters_dict.popitem()
                result = (key, value.numpy())
            except KeyError:
                result = "KeyError"
        elif "setdefault" in input_dict:
            key = input_dict["setdefault"]["key"]
            default_value = input_dict["setdefault"].get("default", None)

            if key in parameters_dict:
                result = parameters_dict[key].numpy()
            else:
                if default_value is not None:
                    parameters_dict[key] = tf.Variable(default_value)
                    result = parameters_dict[key].numpy()
                else:
                    parameters_dict[key] = None
                    result = None
        elif "update" in input_dict:
            update_dict = {}
            for key, value in input_dict["update"]["parameters"].items():
                update_dict[key] = tf.Variable(value)

            parameters_dict.update(update_dict)
            result = {k: v.numpy() for k, v in parameters_dict.items()}
        elif "values" in input_dict:
            values = list(parameters_dict.values())
            result = [v.numpy() for v in values]
        else:
            result = {k: v.numpy() for k, v in parameters_dict.items()}

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        }
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"]["left"], tf_result["result"]["left"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["result"]["right"], tf_result["result"]["right"], atol=A_TOL), "Results do not match"

    input_data_clear = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        },
        "clear": True
    }

    torch_result_clear = torch_version(input_data_clear)
    tf_result_clear = tensorflow_version(input_data_clear)

    assert torch_result_clear["result"] == tf_result_clear["result"], "Clear results do not match"

    input_data_copy = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        },
        "copy": True
    }

    torch_result_copy = torch_version(input_data_copy)
    tf_result_copy = tensorflow_version(input_data_copy)

    assert np.allclose(torch_result_copy["result"]["left"], tf_result_copy["result"]["left"], atol=A_TOL), "Copy results do not match"
    assert np.allclose(torch_result_copy["result"]["right"], tf_result_copy["result"]["right"], atol=A_TOL), "Copy results do not match"

    input_data_fromkeys = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        },
        "fromkeys": {
            "keys": ["new_key1", "new_key2"],
            "default": np.random.rand(3,3).astype(np.float32)
        }
    }

    torch_result_fromkeys = torch_version(input_data_fromkeys)
    tf_result_fromkeys = tensorflow_version(input_data_fromkeys)
    assert np.allclose(torch_result_fromkeys["result"]["new_key1"], tf_result_fromkeys["result"]["new_key1"], atol=A_TOL), "Fromkeys results do not match"

    input_data_get = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        },
        "get": {
            "key": "left",
            "default": np.random.rand(3,3).astype(np.float32)
        }
    }
    torch_result_get = torch_version(input_data_get)
    tf_result_get = tensorflow_version(input_data_get)
    assert np.allclose(torch_result_get["result"], tf_result_get["result"], atol=A_TOL), "Get results do not match"
    
    input_data_items = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        },
        "items": True
    }
    torch_result_items = torch_version(input_data_items)
    tf_result_items = tensorflow_version(input_data_items)
    assert np.allclose(torch_result_items["result"]["left"], tf_result_items["result"]["left"], atol=A_TOL), "Items results do not match"
    assert np.allclose(torch_result_items["result"]["right"], tf_result_items["result"]["right"], atol=A_TOL), "Items results do not match"

    input_data_keys = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        },
        "keys": True
    }
    torch_result_keys = torch_version(input_data_keys)
    tf_result_keys = tensorflow_version(input_data_keys)
    assert torch_result_keys["result"] == tf_result_keys["result"], "Keys results do not match"

    input_data_pop = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        },
        "pop": {
            "key": "left"
        }
    }

    torch_result_pop = torch_version(input_data_pop)
    tf_result_pop = tensorflow_version(input_data_pop)
    assert np.allclose(torch_result_pop["result"], tf_result_pop["result"], atol=A_TOL), "Pop results do not match"

    input_data_setdefault = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
        },
        "setdefault": {
            "key": "right",
            "default": np.random.rand(5, 10).astype(np.float32)
        }
    }

    torch_result_setdefault = torch_version(input_data_setdefault)
    tf_result_setdefault = tensorflow_version(input_data_setdefault)
    assert np.allclose(torch_result_setdefault["result"], tf_result_setdefault["result"], atol=A_TOL), "Setdefault results do not match"
    
    input_data_update = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        },
        "update": {
            "parameters":{
                 "new_key": np.random.rand(5, 10).astype(np.float32)
            }
        }
    }

    torch_result_update = torch_version(input_data_update)
    tf_result_update = tensorflow_version(input_data_update)
    assert np.allclose(torch_result_update["result"]["left"], tf_result_update["result"]["left"], atol=A_TOL), "Update results do not match"
    assert np.allclose(torch_result_update["result"]["new_key"], tf_result_update["result"]["new_key"], atol=A_TOL), "Update results do not match"

    input_data_values = {
        "parameters": {
            "left": np.random.rand(5, 10).astype(np.float32),
            "right": np.random.rand(5, 10).astype(np.float32)
        },
        "values": True
    }

    torch_result_values = torch_version(input_data_values)
    tf_result_values = tensorflow_version(input_data_values)

    assert np.allclose(torch_result_values["result"][0], tf_result_values["result"][0], atol=A_TOL), "Values results do not match"
    assert np.allclose(torch_result_values["result"][1], tf_result_values["result"][1], atol=A_TOL), "Values results do not match"

    print("Success")

if __name__ == "__main__":
    main()