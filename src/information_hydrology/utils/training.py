from enum import Enum

from hy2dl.datasetzoo.camelsus import CAMELS_US
from torch.utils.data import Dataset


class Period(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    
def get_dataset(config: dict, period: Period = Period.TRAINING) -> Dataset:
    validate_data = False
    match period:
        case Period.TRAINING:
            period = config["train_period"]
            validate_data = True
        case Period.VALIDATION:
            period = config["validation_period"]

    ds = CAMELS_US(
        dynamic_input=config["dynamic_inputs"],
        forcing=config["forcings"],
        target=config["target_variables"],
        sequence_length=config["sequence_length"],
        time_period=period,
        path_data=config["data_dir"],
        path_entities=config["train_basin_file"],
        check_NaN=validate_data,
        static_input=config["static_attributes"],
    )
    return ds