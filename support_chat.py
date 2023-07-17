import torch
import transformers
from utils.support_bot_util import load_model_and_tokenizer
from constants import SAVED_MODEL_NAME, get_model_path, MODEL_NAME

# Load saved model and tokenizer
# Specify the path to save or load the model
model_path = get_model_path()

# Load pre-trained model and tokenizer
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, model_path)

# Set the model to evaluation mode
model.eval()

# Define a function to generate a response given user input
def generate_response(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')

    # Generate output response
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response

# Interactive chat loop
while True:
    user_input = input('User: ')
    if user_input.lower() == 'exit':
        break

    # Generate response
    response = generate_response(user_input)
    print('Bot:', response)
