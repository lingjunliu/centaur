For the api {api}, provide a signature of the api as a dictionary. Here are the specifications:
- The dictionary should have three keys: 
    - `args` for positional arguments (parameters). The value should be a dictionary with names of the parameters as the key (e.g. "input"). The values should be the type of argument it accepts.
    - `kwargs` for keyword arguments. The value should be a dictinary similar to the one above.
    - `inner` for apis that are callable modules i.e. they return a function that can be called with more parameters. This will contain the signature required to call the function returned by the api. This value should be another dictionary with `args` and `kwargs` if the api is a callable module. Otherwise, keep this dictionary empty (i.e. do not include the `args` or `kwargs` keys). {callables}
- **IMPORTANT** The types of each parameter can only be (strictly) one of the following options:
    - "tensor"
    - "tensor_list"
    - "integer"
    - "float"
    - "boolean"
    - "string"
    - "dtype"
    - "list"
    - "tuple"
- Choose the value that most closely matches with the argument type. For example, change IntTensor or FloatTensor type arguments to "tensor", change "Numbers" to "float" etc.
- If you are confused about a type, choose the best one and add a small comment on what you think should have been there instead.
- Do not add the argument "device".
- If a parameter name starts with an underscore `_` and/or the document does not describe the parameter (i.e. internal parameters), skip including the parameter in the signature.
- If the api has multiple signatures, add both with a prefix to the dictionary key enumerating the count (e.g. `"{api}_1"`, `"{api}_2"` etc). Same goes for argument types, if an argument can be of multiple types (e.g. tuples and integers), create new signatures for each possible type with same prefixes as mentioned before.
- **IMPORTANT** Assume I already have a dictionary called `signatures` (you don't have to initialize it). Provide the output as assigning the signature to the entry of the `signatures` dictionary. 

Examples:
{examples}

- Only provide the code, skip any other text. Do not include verbose comments inside code other than short comments about confusion. Do not make any system calls within the code. Do not break any dependencies.