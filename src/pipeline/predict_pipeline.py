from src.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline
import torch

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.summarizer = pipeline(
            "summarization",
            model=self.config.model_path,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        self.gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "max_length": 128,
            "min_length": 30,  # optional: control min length of summary
            "early_stopping": True
        }

    def predict(self, text):
        print("Dialogue:")
        print(text)

        output = self.summarizer(text, **self.gen_kwargs)[0]["summary_text"]

        print("\nModel Summary:")
        print(output)

        return output

