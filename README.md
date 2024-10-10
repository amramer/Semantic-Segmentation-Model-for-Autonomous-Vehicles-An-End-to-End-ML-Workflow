# Semantic Segmentation Model for Autonomous Vehicles - An End-to-End ML Workflow

<img src="https://drive.google.com/uc?id=11UKUhXO8XSmjLn-JgnqjOr1iKvGDRE6u" alt="GIF" width="730" height="550">

This project focuses on instance segmentation using the BDD100K dataset, a large-scale, diverse dataset for autonomous driving. The objective is to segment and identify various objects in street scenes, including:

- **Road**
- **Traffic Light**
- **Traffic Sign**
- **Person**
- **Vehicle**
- **Bicycle**
- **Background**

## Key Features

- **Dataset**: BDD100K, widely used for autonomous driving research, contains images captured from diverse driving scenarios.
- **Classes**: The model segments the following seven classes: `['background', 'road', 'traffic light', 'traffic sign', 'person', 'vehicle', 'bicycle']`.
- **Experiment Tracking**: Uses Weights & Biases (W&B) to track and analyze experiments, providing insights into model performance and training efficiency.

## Technologies Used

- **Python**: For scripting and model development.
- **PyTorch**: For building and training the deep learning model.
- **Weights & Biases**: For experiment tracking, logging metrics, and analyzing results.
