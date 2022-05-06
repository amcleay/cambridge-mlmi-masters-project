# Cambridge Masters Project
Joint Learning of Practical Dialogue Systems and User Simulators

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
   pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
   ```