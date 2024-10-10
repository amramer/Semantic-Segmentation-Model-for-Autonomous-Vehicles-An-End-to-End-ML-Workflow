# Semantic Segmentation Model for Autonomous Vehicles - An End-to-End ML Workflow

  <img src="https://github.com/amramer/Semantic-Segmentation-Model-for-Autonomous-Vehicles-An-End-to-End-ML-Workflow/blob/main/final_segmentation.gif" alt="GIF" width="730" height="550">

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

- [Read the report on Weights & Biases](https://api.wandb.ai/links/amribrahim-amer-2024/5xdtb8eg) <svg fill="#ffff00" width="64px" height="64px" viewBox="-2.4 -2.4 28.80 28.80" role="img" xmlns="http://www.w3.org/2000/svg" stroke="#ffff00"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><path d="M2.48 0a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1zm19.04 0a1.55 1.55 0 1 0 0 3.101 1.55 1.55 0 0 0 0-3.101zM12 2.295a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1zM2.48 5.272a2.48 2.48 0 1 0 0 4.96 2.48 2.48 0 0 0 0-4.96zm19.04 0a2.48 2.48 0 1 0 0 4.96 2.48 2.48 0 0 0 0-4.96zM12 8.496a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1zm-9.52 3.907a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1zm19.04 0a1.55 1.55 0 1 0 0 3.102 1.55 1.55 0 0 0 0-3.102zM12 13.767a2.48 2.48 0 1 0 0 4.962 2.48 2.48 0 0 0 0-4.962zm-9.52 3.907a2.48 2.48 0 1 0 .001 4.962 2.48 2.48 0 0 0 0-4.962zm19.04.93a1.55 1.55 0 1 0 0 3.102 1.55 1.55 0 0 0 0-3.101zM12 20.9a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1z"></path></g></svg>
- Alternatively, check the report pdf file included in this repository.

## Key Features

- **Dataset**: BDD100K, widely used for autonomous driving research, contains images captured from diverse driving scenarios.
- **Classes**: The model segments the following seven classes: `['background', 'road', 'traffic light', 'traffic sign', 'person', 'vehicle', 'bicycle']`.
- **Experiment Tracking**: Uses Weights & Biases (W&B) to track and analyze experiments, providing insights into model performance and training efficiency.

## Technologies Used

- **Python**: For scripting and model development.
- **PyTorch**: For building and training the deep learning model.
- **Fastai**: A high-level library built on PyTorch that simplifies model development and training.
- **Weights & Biases**: For experiment tracking, logging metrics, and analyzing results.
