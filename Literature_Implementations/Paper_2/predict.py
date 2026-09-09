import argparse
import torch
from transformers import AutoTokenizer

# Use exact same architecture
from model import SpeakFasterLLM

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    
    print(f"Loading model from {args.model_path} onto {device}...")
    model = SpeakFasterLLM(vocab_size=vocab_size).to(device)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded checkpoint (Epoch: {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f})")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please train the model first by running `python model.py`")
        return
        
    model.eval()
    
    abbrev_text = args.text
    print(f"\nInput Abbreviation: '{abbrev_text}'")
    
    # Preprocess
    seq_len = 16
    x_tokens = tokenizer(abbrev_text, max_length=seq_len, padding='max_length', truncation=True, return_tensors="pt")
    src = x_tokens['input_ids'].to(device)
    
    # For a simple transformer evaluation, we pass the same src as tgt to force generation in this mock
    tgt = src.clone()
    
    with torch.no_grad():
        outputs = model(src, tgt)
        _, preds = torch.max(outputs, 2)
        
    pred_text = tokenizer.decode(preds[0], skip_special_tokens=True)
    
    print("-" * 30)
    print("Prediction Results:")
    print(f"Expanded Text Generated: {pred_text}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Text with Paper 2 LLM")
    parser.add_argument("--model_path", type=str, default="models/paper2_model.pt", help="Path to saved model checkpoint")
    parser.add_argument("--text", type=str, default="h a y", help="Abbreviated text to expand")
    
    args = parser.parse_args()
    predict(args)
