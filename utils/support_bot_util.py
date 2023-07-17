import transformers
import os

def load_model_and_tokenizer(model_name, model_path):
    if os.path.exists(model_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_path)
        print("Loaded model from saved directory")
        try:
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_path)
            print("Loaded tokenizer from saved directory")
        except OSError:
            print(f"Tokenizer not found in the saved directory. Falling back to default tokenizer.")
            raise OSError(f"Failed to load tokenizer from the saved directory: {model_path}")
            #tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
    elif not model_name:
        raise FileNotFoundError(f"Model or tokenizer not found in the saved directory: {model_path}")        
    else:
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
        print(f"Loaded {model_name} model and tokenizer")
    return model, tokenizer