from wine_quality import logger
from wine_quality.pipeline.reporting import ReportingPipeline
from wine_quality.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from wine_quality.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from wine_quality.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)
from wine_quality.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from wine_quality.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


def run_stage(stage_name, pipeline):
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    stages = [
        ("Data Ingestion stage", DataIngestionTrainingPipeline()),
        ("Data Validation stage", DataValidationTrainingPipeline()),
        ("Data Transformation stage", DataTransformationTrainingPipeline()),
        ("Model Trainer stage", ModelTrainerTrainingPipeline()),
        ("Model evaluation stage", ModelEvaluationTrainingPipeline()),
        ("Reporting stage", ReportingPipeline()),
    ]

    for stage_name, pipeline in stages:
        run_stage(stage_name, pipeline)
