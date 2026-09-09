import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Use the exact same architecture as training
from model import OptimizedCNN

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path} onto {device}...")
    
    # Initialize model
    model = OptimizedCNN(num_classes=5).to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded checkpoint (Epoch: {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f})")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please train the model first by running `python model.py`")
        return
        
    model.eval()
    
    # Preprocessing identical to training
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    # If no image provided, generate a random tensor to simulate webcam crop
    if args.image_path:
        try:
            image = Image.open(args.image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            print(f"Predicting on real image: {args.image_path}")
        except Exception as e:
            print(f"Could not load image {args.image_path}: {e}")
            return
    else:
        print("No image provided. Using a random synthetic crop to simulate webcam feed...")
        input_tensor = torch.randn(1, 3, 64, 64).to(device)
        
    classes = ['Open', 'Closed', 'Left', 'Right', 'Blink']
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred_idx = np.argmax(probs)
        
    print("-" * 30)
    print("Prediction Results:")
    print(f"Predicted Eye Sign: {classes[pred_idx]} (Index: {pred_idx})")
    print(f"Confidence: {probs[pred_idx]:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Eye Signs with Paper 3 CNN")
    parser.add_argument("--model_path", type=str, default="models/paper3_model.pt", help="Path to saved model checkpoint")
    parser.add_argument("--image_path", type=str, default=None, help="Path to an input eye image (.jpg/.png)")
    
    args = parser.parse_args()
    predict(args)
