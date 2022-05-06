# crazyneuraluser

A powerful, the best, domain independent, smart and awesome user model for your system development and evaluation.


## Running baselines

### Environment setup 

In order to set up the necessary environment:

1. Create an environment `neuraluserbaselines` with the help of [conda]:
   ```
   conda env create -f baselines_environment.lock.yml
   ```
2. activate the new environment with:
   ```
   conda activate neuraluserbaselines
   ```

> **_NOTE:_**  The conda environment will have `convlab-2` installed in editable mode - it will appear under `src/convlab-2`.

> **_NOTE:_**  For reproducibility, it is essential to use `baselines_environment.lock.yml` as opposed to `baselines_environment.yml` when creating the environment.

> **_NOTE:_** Due to available hardware, we could not run `convlab2` with the `pytorch` version that comes with their installation. We used version `1.7.1` instead. For our hardware we followed the steps:

1. Uninstall `pytorch`
   ```
   pip uninstall pytorch
   ```
2. Install hardware compatible version of `pytorch`
   ```
   pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   ```

If your device is not compatible with the `cuda 11.0` toolkit, then head over to the [pytorch website](https://pytorch.org/get-started/previous-versions/) and find an appropriate command for installing the version of `pytorch` indicated in `baselines_environment.lock.yml` with a cuda version that runs for your hardware. For example for `cuda 10.2` use:
   ```
   pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
   ```
   
### Generating dialogues through agent-agent interaction

To generate dialogues, first change working directory to the `baselines` directory. Run the command
   ```
   python baselines_setup.py
   ```
to prepare `convlab2` for running the baselines. 

#### Generating dialogues conditioned on randomly sampled goals

Select one of the available configurations in the `configs` directory and run the command
   ```
   python simulate_agent_interaction.py --config /rel/path/to/chosen/config
   ```
to generate dialogues conditioned on randomly sampled goals according to the `convlab2` goal model. The dialogues will be be saved automatically in the `models` directory, under a directory whose name depends on the configuration run. The `models` directory is located in the parent directory of the `baselines` directory. The `metadata.json` file saved with the dialogues contains information about the data generation process.

#### Generating dialogues conditioned on `MultiWOZ2.1` goals

To generate the entire corpus, simply pass the `--goals-path /path/to/multiwoz2.1/data.json/file` flag to `simulate_agent_interaction.py`. To generate the `test/val` split additionally pass the `--filter-path /path/to/multiwoz2.1/test-or-valListFile` argument to `simulate_agent_interaction.py`. You can use the  `generate_multiwoz21_train_id_file` function in `baselines/utils.py` to generate `trainListFile` which can then be passed via the `--filter-path` argument to the dialogue generation script in order to generate dialogues conditioned on the `MultiWOZ2.1` training goals.

### Converting the generated dialogues to SGD-like format

The `create_data_from_multiwoz.py` script can be used to convert the generated dialogues to SGD format, necessary for evaluation. It is based on the script provided by Google for DSTC8, but with additional functionality such as:

   - conversion of slot names as annotated in the MultiWOZ 2.1 dialogue acts to different slot names, specified through the `--slots_convention` argument. Options are `multiwoz22` to convert the slots to the same slots as defined in the MultiWOZ 2.2 dataset whreas the `multiwoz_goals` converts the slot names to the names used in the dialogue goal and state tracking annotations.

  - addition of system and user `nlu` fields for every turn

  - option to perform cleaning operations on the goals to ensure a standard format is received by the evaluator. 

The conversion is done according to the `schema.json` file in the `baselines` directory, which is the same as used by `DSTC8` conversion except for the addition of the `police` domain. Type ``python create_data_from_multiwoz.py --helpfull`` to see a full list of flags and usage. 

## Installation

The recommended way to use this repository is to develop the core code under `src/crazyneuraluser`. The experiments/exporatory analysis making use of the core 
package code should be placed outside the library and import it. See more guidance under the [Project Organisation](#project-organization) section below.

To create an environment for the package, make sure you have deactivated all `conda` environments. Then:

1. Create an environment `crazyneuraluser` with the help of [conda]:
   ```
   conda env create -f environment.yml
   ```
2. Add the developer dependencies to this environment with the help of [conda]:
   ```
   conda env update -f dev_environment.yml
   ```
   
Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You _are encouraged_ to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

4. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.

Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n crazyneuraluser -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```
## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build system configuration. Do not change!
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual Python package, e.g. train_model.py.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `pip install -e .` to install for development or
|                              or create a distribution with `tox -e build`.
├── src
│   └── crazyneuraluser     <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.0.1 and the [dsproject extension] 0.6.1.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
