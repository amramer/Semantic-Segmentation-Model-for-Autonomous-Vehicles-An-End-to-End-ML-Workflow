# Semantic Segmentation Model for Autonomous Vehicles - An End-to-End ML Workflow

<img src="https://github.com/amramer/Semantic-Segmentation-Model-for-Autonomous-Vehicles-An-End-to-End-ML-Workflow/blob/main/media/predictions.jpg" alt="JPG" width="840" height="924">

<img src="https://github.com/amramer/Semantic-Segmentation-Model-for-Autonomous-Vehicles-An-End-to-End-ML-Workflow/blob/main/media/final_segmentation.gif" alt="GIF" width="800" height="620">

This project focuses on instance segmentation using the BDD100K dataset, a large-scale, diverse dataset for autonomous driving. The objective is to segment and identify various objects in street scenes, including:

- **Road**
- **Traffic Light**
- **Traffic Sign**
- **Person**
- **Vehicle**
- **Bicycle**
- **Background**

## Want to know more?

If you'd like to learn more about the project and the results, feel free to read the detailed project report:

- [Read the report on Weights & Biases](https://api.wandb.ai/links/amribrahim-amer-2024/5xdtb8eg)
- Alternatively, check the report PDF file included in this repository.

## Key Features

- **Dataset**: [BDD100K](https://www.vis.xyz/bdd100k/), widely used for autonomous driving research, contains images captured from diverse driving scenarios.
- **Classes**: The model segments the following seven classes: `['background', 'road', 'traffic light', 'traffic sign', 'person', 'vehicle', 'bicycle']`.
- **Experiment Tracking**: Use [Weights & Biases (W&B)](https://wandb.ai/site/) to track and analyze experiments, providing insights into model performance and training efficiency.

## Technologies Used

- **Python**: For scripting and model development.
- **PyTorch**: For building and training the deep learning model.
- **Fastai**: A high-level library built on PyTorch that simplifies model development and training.
- **Weights & Biases**: For experiment tracking, logging metrics, and analyzing results.

---

## Getting Started

### Prerequisites

- Make sure you have Python 3.8 or higher installed.
- Install Conda if you haven't already: [Download Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Running the Colab Notebook

To get started with the project, you can use the Colab notebook provided in the repository:

- **Run the notebook**: [Segmentation_Model_Autonomous_Vehicle.ipynb](Segmentation_Model_Autonomous_Vehicle.ipynb)
  
The notebook installs the project dependencies from the `requirements.txt`, starts a new W&B run, downloads and preprocesses the dataset, trains the model, and evaluates it.

### Dataset Preparation

To use the BDD100K dataset, you can download it from the [official BDD100K website](https://www.vis.xyz/bdd100k/).

### Running Locally

You can also run the project locally by following these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/amramer/Semantic-Segmentation-for-Autonomous-Vehicles.git
   cd Semantic-Segmentation-for-Autonomous-Vehicles

2. **Create and activate the Conda enviroment**:
   Make sure you have Conda installed, then run:

   ```bash
   conda env create -f conda-environment.yaml
   conda activate segmentation-env

3. **Run the training script:**

    Use the default hyperparameters by running the training script:

     ```bash
     python train.py
     ```

    To customize hyperparameters, you can use optional arguments. Run the following command to see all available options:
    
    ```bash
      python train.py --help
    ```
    ### Output of `train.py --help:`

    ```bash
    usage: train.py [-h] [--img_size IMG_SIZE] [--batch_size BATCH_SIZE]
                    [--epochs EPOCHS] [--lr LR] [--arch ARCH]
                    [--augment AUGMENT] [--seed SEED]
                    [--log_preds LOG_PREDS] [--pretrained PRETRAINED]
                    [--mixed_precision MIXED_PRECISION]
    
    Process hyper-parameters for training the segmentation model.
    
    optional arguments:
      -h, --help            show this help message and exit
      --img_size IMG_SIZE   image size (default: 512)
      --batch_size BATCH_SIZE
                            batch size (default: 16)
      --epochs EPOCHS       number of training epochs (default: 20)
      --lr LR               learning rate (default: 0.001)
      --arch ARCH           timm backbone architecture (default: resnet34)
      --augment AUGMENT     Use image augmentation (default: True)
      --seed SEED           random seed (default: 42)
      --log_preds LOG_PREDS log model predictions (default: True)
      --pretrained PRETRAINED
                            Use pretrained model (default: True)
      --mixed_precision MIXED_PRECISION
                            use fp16 for mixed precision (default: True)
    ```


      **Examples:**
        
      - Train with a custom image size and batch size:
      
          ```bash
          python train.py --img_size 640 --batch_size 16
      
      - Change the number of epochs and learning rate:
      
          ```bash
          python train.py --epochs 50 --lr 0.0001

   ## Hyperparameter Optimization

   To get the best performance out of the model, we can fine-tune or optimize the set of hyperparameters (such as batch size, learning rate, etc.). Instead 
   of manually setting custom values for these parameters, **W&B Sweeps** allow us to automate the hyperparameter tuning process, efficiently exploring a 
   range of values to find the optimal configuration for our model.
    
   You can refer to the [Weights & Biases Sweep Documentation](https://docs.wandb.ai/guides/sweeps) for more details.

   ### Steps for Hyperparameter Optimization

    1. **Define Sweep Configuration**:
   
       The sweep configuration is defined in the [`sweep.yaml`](sweep.yaml) file. This file contains the settings for the sweep, such as the search method 
       and hyperparameter space.

       Key components of the configuration:
         - **program**: Specifies the script to run (e.g., `train.py`).
         - **method**: We use `random` search to explore different configurations.
         - **metric**: The goal is to maximize the mean Intersection over Union (mIoU) metric.
         - **parameters**: Defines the hyperparameters to tune, such as learning rate, batch size, and [backbone model architecture](https://pytorch.org/vision/stable/models.html).
      
       Example of `sweep.yaml`:
         
         ```yaml
         program: train.py
         method: random
         project: Semantic-Segmentation-Model-for-Autonomous-Vehicle
         entity: av-team
         metric:
           name: miou
           goal: maximize
         parameters:
           lr:
             distribution: log_uniform_values
             min: 1e-5
             max: 1e-2
           batch_size:
             values: [4, 8]
           arch:
             values: ['resnet18', 'convnext_tiny', 'regnet_x_400mf', 'mobilenet_v3_small']```

      2. **Initialize the Sweep**:

         Run the following command to create and initialize the sweep:

         ```bash
          wandb sweep sweep.yaml
         ```
         This will create a new sweep and return a unique sweep ID.

      3. **Launch Agents:**:

         Once the sweep is initialized, you can launch agents to start running the sweep. Each agent runs one instance of the sweep:

         ```bash
         wandb agent <SWEEP_ID>
         ```
         To limit the maximum number of iterations or runs for the sweep per agent, use the following command:

         ```bash
         wandb agent <SWEEP_ID> --count 30
         ```
         Note: You can choose the number of runs by setting the `--count` flag to the desired value.

         To run two W&B agents simultaneously on different GPUs with a set number of runs (--count), use the following commands:

         ```bash
         CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID> --count 30 &
         CUDA_VISIBLE_DEVICES=1 wandb agent <SWEEP_ID> --count 30 &
         ```
         This runs one agent on GPU 0 and another on GPU 1, each performing 30 runs. The `&` allows the commands to run in the background, so both agents run 
         at the same time.
  
         * The parallel coordinate plot below, generated from the W&B sweep, visualizes different runs with varying hyperparameter combinations for         
           architecture, batch size, and learning rate. The resulting mIoU for each run is shown on the right.

         <img src="https://github.com/amramer/Semantic-Segmentation-Model-for-Autonomous-Vehicles-An-End-to-End-ML-Workflow/blob/main/media/sweep-runs.png" 
           alt="PNG" width="1460" height="550">

         



