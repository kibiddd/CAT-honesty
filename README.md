# Continuous-AdvTrain for Honesty

This is a repository for the continuous adversarial training for honesty project .
This repo is built on top of sophie-xhonneux/Continuous-AdvTrain


## Training Model for honesty with CAT

The default data used is in data/sampled_adv_training_fixed_v2.csv, which is crafted using the `provided_fact` subset of the MASK benchmark.

## Running the code

1. Create a config in `config/path` (see `example_path.yaml`)
2. Run the code with `python src/run_experiments.py --config-name=adv_train_ul path=example_path`

You can also run the IPO experiments by replacing `adv_train_ul` with `adv_train_ipo`. Moreover, hydra allows you to override any hyperparameters from the commandline (e.g. add `adversarial.eps=0.075`) or you can create a new config file under the `config` folder. See the paper for the exact hyperparameters.

## Evaluate Honesty with MASK

The data is in the data folder is from the [MASK benchmark](https://github.com/centerforaisafety/mask/tree/main/mask), with `provided_fact` set modified to exclude data sampled for adversarial training.

