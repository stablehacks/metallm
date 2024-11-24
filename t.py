import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from huggingface_hub import login
from config import Config
from utils import check_system_resources

# Initialize FastAPI and check system resources
check_system_resources(Config.MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"API Using Device: {device}")

if Config.HUGGINGFACE_ACCESS_TOKEN:
    login(token=Config.HUGGINGFACE_ACCESS_TOKEN)

use_auth = bool(Config.HUGGINGFACE_ACCESS_TOKEN)

# Tokenizer and model initialization
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_auth_token=use_auth)
fine_tuned_model_path = "./fine-tuned-model-grants"

if os.path.exists(fine_tuned_model_path):
    # Load the fine-tuned model if it already exists
    print("Loading fine-tuned model...")
    model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path).to(device)
else:
    # Load the base model and proceed to fine-tuning
    print("Fine-tuned model not found. Proceeding with training...")
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME, use_auth_token=use_auth).to(device)

    # Load dataset
    dataset = load_dataset('csv', data_files='g2_cleaned.csv')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        output_dir=fine_tuned_model_path,  # Directory for saving model checkpoints
        save_total_limit=2,  # Limits the total number of checkpoints saved
        save_steps=100,  # Save checkpoint every 100 steps
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    # Train the model
    trainer.train()
    trainer.evaluate()
    trainer.save_model(fine_tuned_model_path)  # Save the fine-tuned model

# Now, the model is either loaded (fine-tuned) or freshly trained, and can be used for inference.

