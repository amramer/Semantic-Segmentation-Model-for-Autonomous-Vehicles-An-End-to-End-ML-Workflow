# Importing Libraries
import wandb
import params
import pandas as pd
import torchvision.models as tvmodels
import os, warnings
warnings.filterwarnings('ignore')
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from fastai.vision.augment import *
from sklearn.model_selection import *
from utils import *
from pathlib import Path

def download_data():
    """Download the processed dataset from W&B Artifact."""
    processed_data_at = wandb.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def label_func(fname):
    """Get the label (mask) file path for a given image file."""
    return (fname.parent.parent / "labels") / f"{fname.stem}_mask.png"

def get_df(processed_dataset_dir, is_test=False):
    """Load data split CSV and assign corresponding file paths."""
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    if not is_test:
        df = df[df.Split != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Split == 'valid'
    else:
        df = df[df.Split != 'train'].reset_index(drop=True)
        df['is_valid'] = df.Split == 'valid'
    df["image_fname"] = [processed_dataset_dir / f'images/{f}' for f in df.File_Name.values]
    df["label_fname"] = [label_func(f) for f in df.image_fname.values]
    return df

def count_by_class(arr, cidxs): 
    """Count occurrences of each class."""
    return [(arr == n).sum(axis=(1, 2)).numpy() for n in cidxs]

def log_hist(c):
    """Plot and log histograms of class distribution in target vs predictions."""
    _, bins, _ = plt.hist(target_counts[c], bins=10, alpha=0.5, density=True, label='target')
    _ = plt.hist(pred_counts[c], bins=bins, alpha=0.5, density=True, label='pred')
    plt.legend(loc='upper right')
    plt.title(params.BDD_CLASSES[c])
    img_path = f'hist_val_{params.BDD_CLASSES[c]}'
    plt.savefig(img_path)
    plt.clf()
    wandb.log({img_path: wandb.Image(f'{img_path}.png', caption=img_path)})

def main():
    # Start the W&B run
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="evaluation", tags=['staging'])

    # Download the model artifact from W&B registry
    # Replace this with the model artifact path of your W&B best model 
    artifact = run.use_artifact('amribrahim-amer-2024-org/wandb-registry-model/Autonomous Driving Semantic Segmentation:v0', type='model')
    artifact_dir = Path(artifact.download())
    model_path = artifact_dir.ls()[0].parent.absolute() / artifact_dir.ls()[0].stem

    # Get the model producer run and update W&B config
    producer_run = artifact.logged_by()
    wandb.config.update(producer_run.config)
    config = wandb.config

    # Download processed dataset
    processed_dataset_dir = download_data()
    test_valid_df = get_df(processed_dataset_dir, is_test=True)
    test_valid_dls = get_data(test_valid_df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

    # Set up the model architecture
    arch = config.Learner['arch'] if 'arch' in config.Learner else 'resnet18'
    learn = unet_learner(test_valid_dls, arch=getattr(tvmodels, arch.split('.')[-1]), pretrained=config.pretrained, metrics=metrics)
    learn.load(model_path)

    # Evaluate on validation and test sets
    val_metrics = learn.validate(ds_idx=1)
    test_metrics = learn.validate(ds_idx=0)

    # Log results to W&B
    log_predictions(learn)
    wandb.summary['validation_metrics'] = val_metrics
    wandb.summary['test_metrics'] = test_metrics

    # Log histograms for class counts
    val_probs, val_targs = learn.get_preds(ds_idx=1)
    val_preds = val_probs.argmax(dim=1)
    class_idxs = params.BDD_CLASSES.keys()
    target_counts = count_by_class(val_targs, class_idxs)
    pred_counts = count_by_class(val_preds, class_idxs)
    for c in class_idxs:
        log_hist(c)

    run.finish()

if __name__ == "__main__":
    main()

