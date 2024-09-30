WANDB_PROJECT = "Semantic-Segmentation-Model-for-Autonomous-Vehicle"
ENTITY = None # set this to team name if working in a team
BDD_CLASSES = {i:c for i,c in enumerate(['background', 'road', 'traffic light', 'traffic sign', 'person', 'vehicle', 'bicycle'])}
RAW_DATA_AT = 'dataset'
PROCESSED_DATA_AT = 'dataset_split'
