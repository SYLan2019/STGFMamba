# STGFMamba: Spatio-Temporal Graph Fourier-Enhanced Mamba for Traffic Flow Prediction
## Requirments
We used Python 3.8 and Pytorch 2.1.0 with cudatoolkit 11.7.

This work used the framework code from [https://arxiv.org/abs/2406.11244](https://arxiv.org/abs/2406.11244).
## Usage
We used NVIDIA GeForce RTX 4090 24GB for all our experiments. We provide the template configuration file (`template.json`).

To train STGFMamba, use the `run.py` file as follows:

``` bash
python run.py --config_path=./template.json
```
Results will be printed in the terminal and saved in the directory according to the configuration file.

You can find log files and checkpoints resulting from experiments in the `f"experimental_results/{dataset}-{in_steps}-{out_steps}-{str(train_ratio).zfill(2)}-{seed}-{model}"` directory.

#### Required Packages

```
pytorch
numpy
tqdm
```

```json
{
    "setting": {
        "exp_name": "Name of the experiment.",
        "dataset": "The dataset to be used, e.g., 'pems04'.",
        "model": "The model type to be used, e.g., 'STGFMamba'.",
        "in_steps": "Number of input time steps to use in the model.",
        "out_steps": "Number of output time steps (predictions) the model should generate.",
        "train_ratio": "Percentage of data to be used for training (expressed as an integer out of 100).",
        "val_ratio": "Percentage of data to be used for validation (expressed as an integer out of 100).",
        "seed": "Random seed for the reproducibility of results."
    },
    "hyperparameter": {
        "model": {
            "emb_dim": "Dimension of each embedding.",
            "ff_dim": "Dimension of the feedforward network within the model.",
            "num_layers": "Number of layers in the Transformer encoder.",
            "e_layers": "Number of GFMamba blocks.",
            "order": "Number of elements in the set of Graph Propagate results",
            "dropout": "Dropout rate used in the model."
        },
        "training": {
            "lr_decay_rate": "Decay rate for learning rate.",
            "milestones": [
                "Epochs after which the learning rate will decay."
            ],
            "epochs": "Total number of training epochs.",
            "valid_epoch": "Number of epochs between each validation.",
            "patience": "Number of epochs to wait before early stopping if no progress on the validation set.",
            "batch_size": "Size of the batches used during training.",
            "lr": "Initial learning rate for training.",
            "weight_decay": "Weight decay rate used for regularization during training."
        }
    },
    "cuda_id": "CUDA device ID (GPU ID) to be used for training if available.",
    "force_retrain": "Flag to force the retraining of the model even if a trained model exists."
}
```


