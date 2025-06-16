from dataclasses import dataclass


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    max_context: int = 128
    val_frequency: int = 100
    sample_frequency: int = 500
    temperature: float = 1.0
    metrics_csv_enabled: bool = True
