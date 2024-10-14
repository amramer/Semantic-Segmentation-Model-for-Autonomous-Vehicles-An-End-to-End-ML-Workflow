# Semantic Segmentation Model for Autonomous Vehicles - An End-to-End ML Workflow

  <img src="https://github.com/amramer/Semantic-Segmentation-Model-for-Autonomous-Vehicles-An-End-to-End-ML-Workflow/blob/main/media/predictions.jpg" alt="JPG" width="840" height="924">

  <img src="https://github.com/amramer/Semantic-Segmentation-Model-for-Autonomous-Vehicles-An-End-to-End-ML-Workflow/blob/main/media/final_segmentation.gif" alt="GIF" width="730" height="550">

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
- Alternatively, check the report pdf file included in this repository.

## Key Features

- **Dataset**: [BDD100K](https://www.vis.xyz/bdd100k/), widely used for autonomous driving research, contains images captured from diverse driving scenarios.
- **Classes**: The model segments the following seven classes: `['background', 'road', 'traffic light', 'traffic sign', 'person', 'vehicle', 'bicycle']`.
- **Experiment Tracking**: Use [Weights & Biases (W&B)](https://wandb.ai/site/) to track and analyze experiments, providing insights into model performance and training efficiency.

## Technologies Used

- **Python**: For scripting and model development.
- **PyTorch**: For building and training the deep learning model.
- **Fastai**: A high-level library built on PyTorch that simplifies model development and training.
- **Weights & Biases**: For experiment tracking, logging metrics, and analyzing results.

