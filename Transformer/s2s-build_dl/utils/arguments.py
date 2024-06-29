from dataclasses import dataclass, fields

@dataclass
class TrainingParam:
    checkpoint_path:   str = ""
    experiment_string: str = "test"
    batch_size: int = 40
    grad_accumulation_steps: int = 1
    epochs:     int = 100
    pretrain:  bool = False
    validate:  bool = False
    device:     str = "cpu"
    era5_path:  str = ""
    hrrr_path:  str = ""
    train_file: str = ""
    val_file:   str = ""
    dataparallel: bool = False

    def __repr__(self):

        return_str = "\n"

        for field in fields(self):
            return_str += f"{field.name}\t{getattr(self, field.name)}\n"

        return return_str

@dataclass
class OptParam:
    encoder_learning_rate: float = 1e-3
    encoder_weight_decay:  float = 1e-2
    decoder_learning_rate: float = 1e-3
    decoder_weight_decay:  float = 1e-2

    def __repr__(self):

        return_str = "\n"

        for field in fields(self):
            return_str += f"{field.name}\t{getattr(self, field.name)}\n"

        return return_str

@dataclass
class ModelParam:
    encoder_depth:    int = 8
    encoder_dim:      int = 768
    encoder_channels: int = 12
    encoder_heads:    int = 4
    encoder_mlp_dim:  int = 3*768

    decoder_depth:    int = 8
    decoder_dim:      int = 768
    decoder_channels: int = 2
    decoder_heads:    int = 4
    decoder_mlp_dim:  int = 3*768

    def __repr__(self):

        return_str = "\n"

        for field in fields(self):
            return_str += f"{field.name}\t{getattr(self, field.name)}\n"

        return return_str

