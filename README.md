Invariant-based test input generator for Deep Learning Library APIs, e.g., Jax, Pytorch, and TensorFlow. The goal is to find bugs on these APIs.

The tool is structured in two parts:

 1. Inference of input invariants
 2. Invariant-based input generator

The code is organized as follow:

```markdown
- 📁 bug_reports       # code snippets containing the bugs reported 
- 📁 drivers           # driver to call an API in various libraries
- 📁 eval              # code related to evaluation (e.g. oracle, coverage)
- 📁 generator         # invariant-based input generator
- 📁 invariants        # mined invariants from APIs
- 📁 learner           # invariant learner
- 📁 llm               # code to generate drivers and signatures using gemini
- 📄 pytest.ini        # pytest configuration file
- 📄 requirements.txt  # dependencies of this project
- 📄 torch_apis.txt    # list of supported PyTorch APIs
- 📄 signatures.json   # signatures for the supported APIs
- 📁 scripts           # scripts (e.g., run test, demo generator, etc.)
- 📁 tests             # tests
- 📁 utils             # utility functions
```

<h1>Prerequisites</h1>
 
 - **python**: The tool uses `python 3.12`
 - **venv**: `sudo apt install python3.12-venv`
 - **libopenmp**: `sudo apt-get install libomp-dev`

<h1>Steps to run</h1>

<h2> 1. Learn invariants (offline) </h2>
 
 <h3> Slurm (all variants) </h3>

 To run invariant inference for all variants (variations of the apis from `torch_variations.txt` for PyTorch and `tf_variations.txt` for Tensorflow.), run the following. **Be sure to install and configure slurm before running this.**
 
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/infer_invariants_with_slurm.sh <duration> <regen> <lib>
 ```

 Example:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/infer_invariants_with_slurm.sh 300 1 torch
 ```
 This will generate (regenerate if already exists since `1` is passed as `regen`) the invariants for the variations of apis and it will use a time budget of `300` seconds to do so.

 - `duration`: Max time budget per variation to learn invariants
 - `regen`: 1 to regenerate invariants, 0 to learn invariants only if they do not exist
 - `lib`: `torch` or `tf`
 
 <h3> Without slurm (one variant) </h3>
 To run invariant inference for a single variant, run the following *(under the venv)*:

 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ python -m learner.invariant_inference <variant> <time budget> <1 to regenerate invariants 0 otherwise>
 ```

<h2> 2. Generate models (offline) </h2>
 
 <h3> Slurm (all apis/variants) </h3>

 To generate models by solving the constraints, the script `scripts/generate_models_with_slurm.sh` needs to be used. **Be sure to install and configure slurm before running this.** This runs model generation for all variations of the apis from `torch_variations.txt` for PyTorch and `tf_variations.txt` for Tensorflow. Since this is an offline mode, running this once is enough to run online fuzzing campaigns.

 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/generate_models_with_slurm.sh <duration> <n_max> <lib> <seed> <regen>
 ```
 - `duration`: Time budget for generating model for each api variation in seconds.
 - `n_max`: Passing 0 (default) means no max on number of models. Anything `> 0` will limit the number of models to that number (if it can reach that number before the time budget `duration` runs out).
 - `lib`: `torch` for PyTorch, `tf` for Tenosrflow
 - `seed`: Seed for the generator, default `200`.
 - `regen`: Pass 1 to regenerate models that already exist. Default: 0.

 Example:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/generate_models_with_slurm.sh 3600 1000 torch 42 1
 ```
 This will generate models for each variation of torch apis until 1h passes or 1000 max models are generated, even if models exist. `42` will be used as the seed.
 
 <h3> Without slurm (one variant) </h3>

 To run model generation for one variation or variant (unique signature of an api, full list under `<lib>_variations.txt`) *(under the venv)*:

 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ python -m generator.z3 <variant> <duration> <n_max> <lib> <seed> <regen>
 ```

<h2> 3. Fuzzing (online) </h2>
 
 <h3> Slurm (all apis) </h3>

 To run fuzzing campaings, use the `scripts/fuzz_with_slurm.sh`. **Be sure to install and configure slurm before running this.**. This runs the fuzzing campaign on apis from the file `torch_apis.txt` parallelly.
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/fuzz_with_slurm.sh <duration> <n_max> <lib> <seed>
 ```
 - `duration`: Duration to fuzz each api in seconds.
 - `n_max`: Passing 0 (default) means no max on number of inputs. Anything `> 0` will limit the number of inputs to that number (if it can reach that number before the time budget `duration` runs out).
 - `lib`: `torch` for PyTorch, `tf` for Tenosrflow
 - `seed`: Seed for the generator, default `200`.

 Example:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/fuzz_with_slurm.sh 3600 0 torch 42
 ```
 This will run the `z3` based generator parallelly on all apis in `torch_apis.txt` with `seed=42`, each with a time budget of 1 hour with no limits on the number of inputs or models generated.

 <h3> Without slurm (one api) </h3>

 To fuzz for a single api *(under the venv)*:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ python -m generator.harness_z3 <api> <duration> <n_max> <lib> <seed>
 ```

 <h2> 4. Compute Coverage: Pytorch (evaluation) </h2>
 
 <h3> Slurm (all apis) </h3>

 To compute coverage for all apis, run the following. **Be sure to install and configure slurm before running this.**
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/coverage_with_slurm.sh <n_inputs>
 ```
 - `n_inputs`: Number of inputs per api used for coverage calculation. Passing -1 will cause it to calculate for all inputs.

 <h3> Without slurm (one api) </h3>

 To compute coverage for a single api *(under the venv)*, there are two steps.
 1. Downloading instrumented pytorch (the script above would download it, if that was never run, download it using these commands):
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ pip install gdown
 (venv) ~/dll-fuzzing-with-input-invariants$ gdown --fuzzy https://drive.google.com/file/d/1GqydzvLO7XTlFXnSum_zhEulJpC2JRwU/view?usp=sharing -O instrumented_pytorch/
 ```
 2. Patching:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ python -m eval.patching <api> <n_inputs>
 ```
 3. Coverage:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ pip install instrumented_pytorch/torch*
 (venv) ~/dll-fuzzing-with-input-invariants$ python -m eval.coverage <api>
 ```

 <h2> 5. Run Oracle (bug detection) </h2>
 
 <h3> Slurm (all apis) </h3>

 To run oracle on all apis, run the following. **Be sure to install and configure slurm before running this.**

 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/run_oracle_with_slurm.sh <lib> <low, default: -1> <high, default -1>
 ```
 - low: the index to start running the oracle from. passing -1 will start from the beginning
 - high: the index to run oracle until. passing -1 will go through all inputs.
 
 <h3> Without slurm (one api) </h3>
 To run oracle for a single api *(under the venv)*:

 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ python -m eval.oracle <api> <lib>
 ```

<h1>Random Generation</h1>
 To use random generation instead of the invariant-based approach, run:

 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ python -m generator.random_generation <api> <duration> <n_max> <lib>
 ```

 It will run the code for `duration` seconds unless `n_max` is specified. With `n_max`, it will run until whichever comes first (`duration` seconds or generation of `n_max` inputs)