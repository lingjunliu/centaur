Invariant-based test input generator for Deep Learning Library APIs, e.g., Jax, Pytorch, and TensorFlow. The goal is to find bugs on these APIs.

The tool is structured in two parts:

 1. Inference of input invariants
 2. Invariant-based input generator

The code is organized as follow:

```markdown
- 📁 drivers           # driver to call an API in various libraries
- 📁 generator         # invariant-based input generator
- 📁 invariants        # mined invariants from APIs
- 📁 learner           # invariant learner
- 📄 pytest.ini        # pytest configuration file
- 📄 requirements.txt  # dependencies of this project
- 📄 apis.txt          # list of supported APIs
- 📄 signatures.json   # signatures for the supported APIs
- 📁 scripts           # scripts (e.g., run test, demo generator, etc.)
- 📁 tests             # tests
- 📁 utils             # utility functions
```

<h1>Use cases</h1>

<h2> 1. Learn invariants </h2>

 The `infer_invariants` function in the file `learner/invariant_inference.py` can generate a list of inputs randomly, check which of them are valid and for each valid input, check which rules are satisfied by them. It returns a set of tuples `(rule_name, arg1, arg2, ...)` where `arg1`, `arg2`, ... are the arguments in the input that are relevant for a rule (each rule in our rule set is applicable for pair of arguments).

 The tests written under `tests/test_invariants.py` demonstrates usage of this function.

<h2> 2. Generate tests </h2>

@Abid, can you please complete this. You can do this through tests
