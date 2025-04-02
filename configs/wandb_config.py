from dataclasses import dataclass

@dataclass
class WandbConfig:
    project: str = "iNaturalist-CNN"
    entity: str = None  # Optional: your Wandb username
    group: str = None  # Optional: for organizing runs
    job_type: str = "train"
    tags: list = None
    
    # Sweep configuration
    sweep_config: dict = None