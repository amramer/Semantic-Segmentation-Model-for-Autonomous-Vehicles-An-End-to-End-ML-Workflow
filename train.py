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

# train configurations
train_config = SimpleNamespace(
    framework="fastai",
    img_size=180,
    batch_size=8, 
    augment=True, # use data augmentation
    epochs=50, 
    lr=2e-3,
    pretrained=True,  # whether to use pretrained encoder,
    mixed_precision=True, # use automatic mixed precision
    arch="resnet18",
    seed=42,
    log_preds=False,
)


def parse_args():
    """Overriding default argments"""
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=train_config.img_size, help='image size')
    argparser.add_argument('--batch_size', type=int, default=train_config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=train_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=train_config.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=train_config.arch, help='timm backbone architecture')
    argparser.add_argument('--augment', type=t_or_f, default=train_config.augment, help='Use image augmentation')
    argparser.add_argument('--seed', type=int, default=train_config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=t_or_f, default=train_config.log_preds, help='log model predictions')
    argparser.add_argument('--pretrained', type=t_or_f, default=train_config.pretrained, help='Use pretrained model')
    argparser.add_argument('--mixed_precision', type=t_or_f, default=train_config.mixed_precision, help='use fp16')
    args = argparser.parse_args()
    vars(train_config).update(vars(args))
    return

def download_data():
    """Grab dataset from artifact"""
    processed_data_at = wandb.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def label_func(fname):
    "Get the label (mask) file path for a given image file"
    return (fname.parent.parent/"labels")/f"{fname.stem}_mask.png"
        
def get_df(processed_dataset_dir, is_test=False):
    """
    Load the data split CSV and assign corresponding file paths.
    
    Parameters:
    - processed_dataset_dir: Path to the dataset directory.
    - is_test: Boolean to indicate whether to use test split or train/valid.
    
    Returns:
    - Updated dataframe with image and label paths.
    """
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    
    if not is_test:
        # Use train and validation data, exclude test data
        df = df[df.Split != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Split == 'valid'
    else:
        # Use validation and test data, exclude training data
        df = df[df.Split != 'train'].reset_index(drop=True)
        df['is_valid'] = df.Split == 'valid'

    # Assign image and label file paths to dataframe
    df["image_fname"] = [processed_dataset_dir/f'images/{f}' for f in df.File_Name.values]
    df["label_fname"] = [label_func(f) for f in df.image_fname.values]
    return df

def get_data(df, bs=4, img_size=180, augment=True):
    """
    Create DataLoaders for training, validation, or testing.
    
    Parameters:
    - df: Dataframe containing image and label paths.
    - bs: Batch size for loading data.
    - img_size: Image dimensions (height), can be a single integer or list.
    - augment: Boolean to apply data augmentation during training.
    
    Returns:
    - DataLoaders object.
    """
    # Ensure img_size is an integer
    if isinstance(img_size, list):
        img_size = img_size[0]  # If a list, take the first element
    
    block = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=params.BDD_CLASSES)),
        get_x=ColReader("image_fname"),
        get_y=ColReader("label_fname"),
        splitter=ColSplitter(),
        item_tfms=Resize((img_size, int(img_size * 16 / 9))),  # Resize maintaining aspect ratio
        batch_tfms=aug_transforms() if augment else None  # Apply augmentation if specified
    )
    return block.dataloaders(df, bs=bs)


def log_predictions(learn):
    """Log model predictions and metrics to W&B as a table."""
    samples, outputs, predictions = get_predictions(learn)
    table = create_iou_table(samples, outputs, predictions, params.BDD_CLASSES)
    wandb.log({"val_pred_table": table})
    
def final_metrics(learn):
    """Log latest metrics values"""
    scores = learn.validate()
    metric_names = ['final_loss'] + [f'final_{x.name}' for x in learn.metrics]
    final_results = {metric_names[i] : scores[i] for i in range(len(scores))}
    for k,v in final_results.items(): 
        wandb.summary[k] = v

def train(config):
    # Setting the seed
    set_seed(config.seed)
    # Intializing a W&B run for tracking the experiment and passing the training configuration
    run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training", config=config)
        
    # good practice to inject params using sweeps
    config = wandb.config

    # prepare data
    processed_dataset_dir = download_data()
    proc_df = get_df(processed_dataset_dir)
    dls = get_data(proc_df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

    # Defining metrics for evaluating the model
    metrics = [MIOU(), BackgroundIOU(), RoadIOU(), TrafficLightIOU(),
               TrafficSignIOU(), PersonIOU(), VehicleIOU(), BicycleIOU()]

    cbs = [WandbCallback(log_preds=False, log_model=True), 
           SaveModelCallback(fname=f'run-{wandb.run.id}-model', monitor='miou')]
    cbs += ([MixedPrecision()] if config.mixed_precision else [])

    # Creating a U-Net model with a ResNet18 backbone, using the pre-trained model if specified
    learn = unet_learner(dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, 
                         metrics=metrics)
    # Training the model
    learn.fit_one_cycle(config.epochs, config.lr, cbs=cbs)
    
    # Logging model preditions as table 'if log_preds = True'
    if config.log_preds:
        log_predictions(learn)
    # Logging metrics
    final_metrics(learn)
    
    # Finish the W&B run
    wandb.finish()

if __name__ == '__main__':
    parse_args()
    train(train_config)


