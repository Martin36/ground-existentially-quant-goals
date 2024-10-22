# Learning to Ground Existentially Quantified Goals
This is the code for the paper "Learning to Ground Existentially Quantified Goals" published in *Proceedings of the Twenty-First International Conference on Principles of Knowledge Representation and Reasoning (KR 2024)*

## Cite
Use the following citation to cite the paper:

```
@inproceedings{funkquist-et-al-kr2024,
  author       = {Martin Funkquist and Simon St{\aa}hlberg and Hector Geffner},
  title        = {Learning to Ground Existentially Quantified Goals},
  booktitle    = {Proceedings of the Twenty-First International Conference on Principles
    of Knowledge Representation and Reasoning (KR 2024)},
  publisher    = {IJCAI Organization},
  year         = 2024,
}
```

## Installation
Use Python version 3.10

#### Note: Using Python version 3.11 will not work, as the Pytorch Lightning version used in this project is not compatible with Python 3.11.

Create a virtual environment of your choice and install the requirements:
```
pip install -r requirements.txt
```

If you just want to train/test models, you can install the requirements from the `requirements-min.txt` file, which excludes the packages required for data generation: 

```
pip install -r requirements-min.txt
```

### Using conda
If you are using conda, you can follow these guidelines:

Create a new conda environment:
```
conda create --name ground-quant-goals python=3.10
conda activate ground-quant-goals
```

Then install the requirements:
```
pip install -r requirements.txt
```

## Generate data

#### Note: Before you try to generate data, make sure that you have downloaded the required PDDL files. These can be found here: [https://zenodo.org/records/13235160](https://zenodo.org/records/13235160)

The data generation code expects the pddl files to be in specific directories, so after you download the PDDL files, create a new directory called `data` and put all the directories there. The file structure should look like this:
```
data
├── blocks_1
│   ├── domain.pddl
├── delivery
│   ├── domain.pddl
├── gripper-adj-3
│   ├── domain.pddl
├── visitall
│   ├── domain.pddl
```

### Generate train/validation data
An example of how to generate the value prediction data is in `scripts/generate_val_pred_data.sh`. This script will generate a dataset for Blocks with the hyperparameters provided in the paper. Data for other domains can be generated in a similar way to the example, using arguments as described below. 

```
bash scripts/generate_val_pred_data.sh
```

### Generate optimal test data
An example of how to generate the optimal test data for Blocks can be found in the script `scripts/generate_val_sub_opt_data.sh`. This script will generate a dataset for Blocks with the hyperparameters provided in the paper. 

Run the script:

```
bash scripts/generate_val_sub_opt_data.sh
```

Below is a reference table to generate the optimal test data for the different domains according to the parameters described in the paper.

| Argument            | Blocks-C | Blocks | Gripper  | Delivery | Visitall |
| ---                 | ---      | ---    | ---      | ---      | ---      |
| `--domain`          | blocks   | blocks | gripper  | delivery | visitall |
| `--goal_pred`       | clear    | on     | -        | -        | -        |
| `--nr_goals`        | 6        | 5      | 6        | 6        | 6        |
| `--max_nr_objects`  | 9        | 9      | 15       | 35       | 20       |
| `--max_nr_vars`     | 6        | 6      | 6        | 6        | 6        |
| `--nr_obj_split`    | 7,8,9    | 7,8,9  | 11,13,15 | 25,30,35 | 15,16,20 |

`--recolor_states` should be set in order to have multiple constants of the same color and `--nr_colors=6` should be set for all domains. `--split_sizes=0,500,0` should be for all domains for the optimal test data and `--split_sizes=39500,500,0` for the value prediction data.

## Train the model
To train a model run the script:

```
bash scripts/train_model.sh
```

This script will run the training on the previously created dataset (blocks with size of 40,000). If you downloaded the data, this dataset should also be available. 

## Test the model, optimal dataset
In order to test the model, you need to have the model ID of the model that you just trained. This can be found in the `logs/val_pred` folder.

To test the model, run the following: 

```
PYTHONPATH=src python src/test.py \
  --model_type=val_sub \
  --data_file=data/datasets/test/blocks-on/6v-9m-500_mc/dev.json \
  --models_folder=models/blocks-on/checkpoints
```

Replace the `--models_folder` argument with the path to your own model if you rather want to use that. You can also run just a single checkpoint by using the `--model_path` argument instead of `--models_folder` e.g. `--model_path=models/blocks-on/checkpoints/epoch=2579-step=12900.ckpt`.


## Test the model, LAMA setup
Before you can run LAMA, you need to get Fast Downward. You can do that following the instructions here: https://www.fast-downward.org/QuickStart. Get the .sif file using apptainer. To follow the steps below, you will need the file `fast-downward.sif` located in the main directory of the project.

If you have Apptainer installed, you can simply get the .sif Fast Downward file by running the following command:

```
apptainer pull fast-downward.sif docker://aibasel/downward:latest
```

To test the model in the LAMA setup, you need to provide the path to a specific checkpoint of a trained model. Here is an example for the given blocks model:

```
bash scripts/test_model_lama.sh models/blocks-on/checkpoints/epoch=2579-step=12900.ckpt
```


