import pandas as pd

from wine_quality.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        data = pd.read_csv(self.config.unzip_data_dir)
        data_columns = list(data.columns)
        schema_columns = list(self.config.all_schema.keys())
        validation_status = data_columns == schema_columns

        with open(self.config.status_file, "w") as f:
            f.write(f"Validation status: {validation_status}")

        return validation_status
