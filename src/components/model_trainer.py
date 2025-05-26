from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from src.entity import ModelTrainerConfig
import os
import torch

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load BART-base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # Load dataset from disk
        dataset = load_from_disk(self.config.data_path)

     
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            warmup_steps=0,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,  # Effective batch size = 16
            weight_decay=0.01,
            logging_steps=50,
            evaluation_strategy='steps',
            eval_steps=200,
            save_steps=500,
            save_total_limit=2,
            logging_dir="./logs"
        )

    


        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
        )
        
        trainer.train()

        # Save model and tokenizer
        model.save_pretrained(os.path.join(self.config.root_dir, "bart-summarizer-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))