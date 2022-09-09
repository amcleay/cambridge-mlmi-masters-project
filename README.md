# Cambridge Masters in Machine Learning Project

Joint Learning of Practical Dialogue Systems and User Simulators

## Dissertation

The dissertation for this project can be read [here](https://static1.squarespace.com/static/60bf1b5023f4f279ec0cf558/t/62fe43cdb9986b33de522282/1660830683789/Alistair+McLeay+Cambridge+Masters+Thesis.pdf).

## Demo

To interact with the dialogue system and user simulator, go to https://huggingface.co/spaces/alistairmcleay/cambridge-masters-project.

## Acknowledgements

This code was developed using [the code and model](https://github.com/TonyNemo/UBAR-MultiWOZ) published with the AAAI 2021 paper "UBAR: Towards Fully
End-to-End Task-Oriented Dialog System with GPT-2", the [gpt2-user-model](https://github.com/andy194673/gpt2-user-model) codebase developed by Andy Tseng while he was a PhD student at Cambridge University, and  These two codebases were critical to this research, and we are very grateful that they chose to open source their work.
 
I would also like to acknowledge the [TRL library](https://lvwerra.github.io/trl/) which was highly valuable, and Huggingface for their Transformers implementation.

Finally, thank you to Professor [Bill Byrne](https://sites.google.com/view/bill-byrne/), [Alex Coca](https://github.com/alexcoca), and [Andy Tseng](https://github.com/andy194673) for their support throughout this project.

## Environment setup

1. Create an environment `crazyneuraluser` with the help of [conda]
   ```
   conda env create -f environment.yml
   ```
2. Activate the new environment with:
   ```
   conda activate crazyneuraluser
   ```
3. Install a version of `pytorch` compatible with your hardware (see the [pytorch website](https://pytorch.org/get-started/previous-versions/)). E.g.:
   ```
   pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
   ```

4. Install `spacy` and download the tokenization tool in spacy:
   ```
   pip install spacy'
   python -m spacy download en_core_web_sm
   ```

## Installation

The recommended way to use this repository is to develop the core code under `src/crazyneuraluser`. The experiments/exporatory analysis making use of the core package code should be placed outside the library and imported. See more guidance under the [Project Organisation](#project-organization) section below.

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
