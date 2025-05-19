import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    modules = input_dict.get("modules", None)
    if modules is not None:
        modules = [torch.nn.Linear(10, 10) for _ in range(len(modules))]
    
    module_list = torch.nn.ModuleList(modules)

    if not cpu:
        module_list = module_list.cuda()

    if "append" in input_dict:
        module = torch.nn.Linear(10, 10)
        if not cpu:
            module = module.cuda()
        module_list.append(module)

    if "extend" in input_dict:
        modules_to_extend = [torch.nn.Linear(10, 10) for _ in range(len(input_dict["extend"]))]
        if not cpu:
            modules_to_extend = [m.cuda() for m in modules_to_extend]
        module_list.extend(modules_to_extend)
        
    if "insert" in input_dict:
        module = torch.nn.Linear(10, 10)
        index = input_dict["insert"][0]
        if not cpu:
            module = module.cuda()
        module_list.insert(index, module)

    result = []
    for module in module_list:
        result.append(str(module))

    if not cpu:
        module_list = module_list.cpu()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        modules = input_dict.get("modules", None)
        module_list = []
        if modules is not None:
            for _ in range(len(modules)):
                module_list.append(tf.keras.layers.Dense(10, activation='linear'))

        if "append" in input_dict:
            module_list.append(tf.keras.layers.Dense(10, activation='linear'))

        if "extend" in input_dict:
            for _ in range(len(input_dict["extend"])):
                module_list.append(tf.keras.layers.Dense(10, activation='linear'))

        if "insert" in input_dict:
            module = tf.keras.layers.Dense(10, activation='linear')
            index = input_dict["insert"][0]
            module_list.insert(index, module)
    
        result = []
        for module in module_list:
            result.append(str(module))

        return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "modules": [1, 2, 3],
        "append": True,
        "extend": [4, 5],
        "insert": [1]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert len(torch_result["result"]) == len(tf_result["result"]), "Lengths do not match"

    print("Success")

if __name__ == "__main__":
    main()