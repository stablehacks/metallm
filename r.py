import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from config import Config
from utils import check_system_resources

# Initialize FastAPI and check system resources
app = FastAPI()
check_system_resources(Config.MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"API Using Device: {device}")

if Config.HUGGINGFACE_ACCESS_TOKEN:
    login(token=Config.HUGGINGFACE_ACCESS_TOKEN)

use_auth = bool(Config.HUGGINGFACE_ACCESS_TOKEN)

# Tokenizer and model initialization
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_auth_token=use_auth)
fine_tuned_model_path = "./fine-tuned-model-grants"

# Check if fine-tuned model exists and load it
if os.path.exists(fine_tuned_model_path):
    print("Loading pre-trained model from fine-tuned-model-grants...")
    model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path).to(device)
else:
    raise FileNotFoundError(f"Pre-trained model not found at {fine_tuned_model_path}")

# The model is now loaded and ready for inference.

# Example usage:
input_text = "Your example input here"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate output (you can adjust max_length, temperature, etc.)
with torch.no_grad():
    output = model.generate(input_ids, max_length=50)

# Decode the generated tokens
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text: ", generated_text)
