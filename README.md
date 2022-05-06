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
   pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
   ```

4. Install a version of `spacy` compatible with your hardware (see the [spacy website](https://spacy.io/usage)). E.g.:
   ```
   pip install -U 'spacy[cuda113]'
   python -m spacy download en_core_web_sm
   ```

