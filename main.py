from src.pipeline.train_pipeline import DataTrainingPipeline
from src.logger import logging


STAGES = {
    "Stage 01 - Data Ingestion": "stage_01",
    "Stage 02 - Data Validation": "stage_02",
    "Stage 03 - Data Transformation": "stage_03",
    "Stage 04 - Model Training": "stage_04",
    "Stage 05 - Model Evaluation": "stage_05",
}


if __name__ == "__main__":
    pipeline = DataTrainingPipeline()

    for stage_name, method_name in STAGES.items():
        try:
            logging.info(f">>>>>> Starting {stage_name} <<<<<<")
            getattr(pipeline, method_name)()
            logging.info(f">>>>>> Completed {stage_name} <<<<<<\n")
        except Exception as e:
            logging.exception(f"!!!!! Error occurred in {stage_name}: {e}")
            raise e
