from pathlib import Path

from wine_quality import logger
from wine_quality.components.data_transformation import DataTransformation
from wine_quality.configuration.configuration import ConfigurationManager

STAGE_NAME = "Data Transformation stage"


class DataTransformationTrainingPipeline:
    def main(self):
        with open(Path("artifacts/data_validation/status.txt")) as f:
            status = f.read().split(" ")[-1]

        if status == "True":
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.train_test_spliting()
        else:
            raise Exception("Your data schema is not valid")


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
