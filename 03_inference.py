import torch  # PyTorch library for building and using neural networks
import json  # Library for handling JSON data
import os  # Library for interacting with the operating system
from transformers import DistilBertTokenizer, DistilBertModel  # Pre-trained model and tokenizer from Hugging Face

# Define the maximum sequence length for tokenized input
MAX_LEN = 512

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        """
        Initializes the DistilBERT-based classification model.
        """
        super().__init__()  # Call the parent constructor
        # Load the pre-trained DistilBERT model
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Add a fully connected layer for feature transformation
        self.pre_classifier = torch.nn.Linear(in_features=768, out_features=768)
        # Dropout layer to prevent overfitting
        self.dropout = torch.nn.Dropout(0.3)
        # Final classification layer with 4 output classes
        self.classifier = torch.nn.Linear(in_features=768, out_features=4)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for model inference.
        """
        # Get hidden states from DistilBERT model
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]  # Extract token embeddings
        pooler = hidden_state[:,0]  # Extract [CLS] token representation
        pooler = self.pre_classifier(pooler)  # Apply linear transformation
        pooler = torch.nn.ReLU()(pooler)  # Apply ReLU activation
        pooler = self.dropout(pooler)  # Apply dropout
        output = self.classifier(pooler)  # Compute class scores
        return output

def model_fn(model_dir):
    """
    Load the trained model from a specified directory.
    """
    print("Loading model from:", model_dir)
    model = DistilBERTClass()  # Instantiate the model
    # Load the trained model weights
    model_state_dict = torch.load(os.path.join(model_dir, 'pytorch_distilbert_news.bin'), map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)  # Load state dict into model
    return model

def input_fn(request_body, request_content_type):
    """
    Preprocess the incoming request data.
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)  # Parse JSON request
        sentence = input_data['inputs']  # Extract input text
        return sentence
    else: 
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Perform inference using the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability
    model.to(device)  # Move model to the appropriate device
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  # Load tokenizer
    inputs = tokenizer(input_data, return_tensors="pt").to(device)  # Tokenize input
    
    ids = inputs['input_ids'].to(device)  # Get input IDs
    mask = inputs['attention_mask'].to(device)  # Get attention mask
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(ids, mask)  # Get model predictions
    
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()  # Compute softmax probabilities
    class_names = ["Business", "Science", "Entertainment", "Health"]  # Define class labels
    predicted_class = probabilities.argmax(axis=1)[0]  # Get class index
    predicted_label = class_names[predicted_class]  # Get class label
    
    return {'predicted_label': predicted_label, 'probabilities': probabilities.tolist()}

def output_fn(prediction, accept):
    """
    Convert the model output into a format suitable for end-users.
    """
    if accept == 'application/json':
        return json.dumps(prediction), accept  # Convert prediction to JSON
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
        








    
    






























            