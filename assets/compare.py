
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Paths to the base model and the fine-tuned model
base_model_name = 'meta-llama/Llama-3.1-8B-Instruct'  # Replace with your base model name or path
fine_tuned_model_path = './'

# Load the base model and tokenizer
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the fine-tuned model and tokenizer
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)

# Example input text
#input_text = "Once upon a time"
input_text = "Wht are the symptoms of diabetes?"

# Tokenize the input
base_inputs = base_tokenizer(input_text, return_tensors='pt')
fine_tuned_inputs = fine_tuned_tokenizer(input_text, return_tensors='pt')

# Generate outputs
with torch.no_grad():
    base_output = base_model.generate(**base_inputs, max_length=50)
    fine_tuned_output = fine_tuned_model.generate(**fine_tuned_inputs, max_length=50)

# Decode and print the outputs
base_text = base_tokenizer.decode(base_output[0], skip_special_tokens=True)
fine_tuned_text = fine_tuned_tokenizer.decode(fine_tuned_output[0], skip_special_tokens=True)

# Print the results
print("Base Model Output:")
print(base_text)
print("\nFine-Tuned Model Output:")
print(fine_tuned_text)

