import argparse
import pandas as pd
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from utils.support_bot_util import load_model_and_tokenizer
from constants import SAVED_MODEL_NAME, get_model_path, MODEL_NAME


# Parse command line arguments
parser = argparse.ArgumentParser(description='Script to train a model.')
input_group = parser.add_argument_group('Input source')
input_group.add_argument('--input', type=str, default='command', choices=['command', 'csv'], help='Input source: command or csv')
csv_group = parser.add_argument_group('CSV file options')
csv_group.add_argument('--csv_file', type=str, default=None, help='Path to CSV file')
args = parser.parse_args()

# Simulated user interactions
interactions = []

if args.input == 'command':
    # Prompt user for interactions
    while True:
        input_text = input('User input: ')
        output_text = input('Bot response: ')
        if not input_text or not output_text:
            break
        interaction = {'input': input_text, 'output': output_text}
        interactions.append(interaction)
else:
    # Load interactions from CSV file
    if args.csv_file is None:
        parser.error('Please provide the path to the CSV file using --csv_file argument.')
    try:
        df = pd.read_csv(args.csv_file)
        interactions = df.to_dict('records')
    except FileNotFoundError:
        parser.error(f'CSV file "{args.csv_file}" not found.')

print(interactions[0].get('input', ''))
print(interactions[0].get('output', ''))

# Load pre-trained model and tokenizer
# Specify the path to save or load the model
model_path = get_model_path()

# Load pre-trained model and tokenizer
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, model_path)

# Tokenize and encode user interactions
def tokenize_and_encode_interactions(interactions):
    encoded_interactions = []
    for interaction in interactions:
        input_text = interaction.get('input', '')
        output_text = interaction.get('output', '')

        # Tokenize and encode user input and output
        input_ids = tokenizer.encode(input_text, add_special_tokens=True)
        output_ids = tokenizer.encode(output_text, add_special_tokens=True)

        encoded_interaction = {
            'input_ids': input_ids,
            'output_ids': output_ids
        }
        encoded_interactions.append(encoded_interaction)
    return encoded_interactions

encoded_interactions = tokenize_and_encode_interactions(interactions)

# Pad the sequences in the batch
padded_input_ids = pad_sequence([torch.tensor(interaction['input_ids']) for interaction in encoded_interactions], batch_first=True, padding_value=0)
padded_output_ids = pad_sequence([torch.tensor(interaction['output_ids']) for interaction in encoded_interactions], batch_first=True, padding_value=0)

# Set model to training mode
model.train()

# Fine-tuning hyperparameters
epochs = 3
batch_size = 7
learning_rate = 1e-5

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    num_samples = len(padded_input_ids)
    num_batches = num_samples // batch_size

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size

        input_batch = padded_input_ids[start_index:end_index]
        output_batch = padded_output_ids[start_index:end_index]

        # Set gradients to zero
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_batch, labels=output_batch)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

print("Training complete.")

# Save the trained model
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
