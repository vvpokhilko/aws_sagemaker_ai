# Import necessary libraries
import torch  # PyTorch for deep learning
import numpy as np  # NumPy for numerical operations
from torch.utils.data import DataLoader, Dataset  # For dataset handling
from transformers import DistilBertTokenizer, DistilBertModel  # Transformer model and tokenizer
from tqdm import tqdm  # Progress bar for loops
import argparse  # Argument parsing for command-line execution
import os  # OS operations
import pandas as pd  # DataFrame manipulation

# Define the S3 path where the dataset is stored
s3_path = 's3://hugging-face-multiclass-textclassification-bucket369/training_data/newsCorpora.csv'

# Load dataset from the specified path
# Dataset contains news headlines categorized into different labels
df = pd.read_csv(s3_path, sep='\t', names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])

# Keep only the necessary columns: TITLE (headline) and CATEGORY (label)
df = df[['TITLE','CATEGORY']]

# Mapping dictionary to convert category abbreviations into full category names
my_dict = {
    'e':'Entertainment',
    'b':'Business',
    't':'Science',
    'm':'Health'
}

def update_cat(x):
    """Function to map category abbreviations to full names."""
    x = my_dict[x]
    return x

# Apply category mapping to the dataset
df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_cat(x))
print(df)  # Print dataset (will be logged in CloudWatch for AWS Lambda execution)

# Reduce dataset size to 5% for faster execution (useful during debugging)
df = df.sample(frac=0.05, random_state=369)
df = df.reset_index(drop=True)

# Create an encoding dictionary to assign unique numeric IDs to category labels
encode_dict = {}

def encode_cat(x):
    """Function to convert category labels into unique numeric IDs."""
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x] 

# Apply encoding function to add a numeric category column
df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))
df

# Load the tokenizer for DistilBERT
# Tokenizer will convert text into numerical token sequences for the model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class NewsDataset(Dataset):
    """Custom PyTorch Dataset class for news headlines."""
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.iloc[index, 0]) # more elegant way to access the TITLE column. It uses integer-based indexing meaning index is treated as a position in the df. 
        # 0 selects the row at the specified index, TITLE column
        # title = str(self.data.TITLE[index])
        title = " ".join(title.split())

        # Tokenize the title
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.max_len,
            padding='max_length',  # Pad shorter sequences
            return_token_type_ids=True,
            truncation=True  # Truncate longer sequences
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids':torch.tensor(ids,dtype=torch.long),
            'mask':torch.tensor(mask,dtype=torch.long),
            # 'targets':torch.tensor(self.data.ENCODE_CAT[index],dtype=torch.long),
            'targets':torch.tensor(self.data.iloc[index, 2], dtype=torch.long), # same thing with iloc here, 2 means we access column 3
        }

    def __len__(self):
        return self.len
        
#Splitting the dataset into training (80%) and testing (20%)
train_size = 0.8
train_dataset = df.sample(frac=train_size,random_state=369)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
train_dataset.reset_index(drop=True)

# Print dataset sizes
print("Full dataset: {}".format(df.shape))
print(f"Train dataset: {train_dataset.shape}")
print("Test dataset: {}".format(test_dataset.shape))

# Define training parameters
MAX_LEN = 512  # Maximum sequence length
TRAIN_BATCH_SIZE = 4  # Batch size for training
VALID_BATCH_SIZE = 2  # Batch size for validation

# Create dataset instances for training and testing
training_set = NewsDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = NewsDataset(test_dataset, tokenizer, MAX_LEN)

# Initializing DataLoader parameters
train_parameters = {
    'batch_size':TRAIN_BATCH_SIZE,
    'shuffle':True,
    'num_workers':0
}
test_parameters = {
    'batch_size':VALID_BATCH_SIZE,
    'shuffle':True,
    'num_workers':0 
}

# Create DataLoader instances for training and testing
trainig_loader = DataLoader(training_set,**train_parameters)
testing_loader = DataLoader(testing_set,**test_parameters)

class DistilBERTClass(torch.nn.Module):
    """DistilBERT model for text classification."""
    def __init__(self):
        # super(DistilBERTClass,self).__init__() # This is equivalent to the line below
        super().__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(in_features=768, out_features=768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(in_features=768, out_features=4)
        
    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0] # Focusing on particular output item from the model
        pooler = hidden_state[:,0] # Pooling CLS or Classification token
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def calculate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    # print(big_idx==targets) # tensor ([False, False, True, True])
    # print((big_idx==targets).sum()) # tensor(2)
    # print((big_idx==targets).sum().item()) # 2
    return n_correct

def train(epoch, model, device, training_loader, optimizer, loss_function): 
    """
    Trains the DistilBERT model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): The DistilBERT model being trained.
        device (torch.device): Device on which training is performed (CPU/GPU).
        training_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model parameters.
        loss_function (torch.nn.Module): Loss function used for training.
    """

    tr_loss = 0  # Accumulate total training loss
    n_correct = 0  # Track number of correct predictions
    nb_tr_steps = 0  # Count total training steps
    nb_tr_examples = 0  # Count total training examples

    model.train()  # Set model to training mode

    # Iterate over training batches
    for _, data in enumerate(training_loader, 0):
        # Move input tensors to the designated device (CPU/GPU)
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        # Forward pass: Get model predictions
        outputs = model(ids, mask)  # Call the forward method

        # Compute loss
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()  # Accumulate total loss

        # Get the predicted class with the highest probability
        big_val, big_idx = torch.max(outputs.data, dim=1)

        # Compute number of correct predictions
        n_correct += calculate_accu(big_idx, targets)

        # Update batch step and example counters
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        # Log training loss and accuracy every 5000 steps
        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        # Zero out previous gradients to prevent accumulation
        optimizer.zero_grad()

        # Backpropagation: Compute gradients of the loss with respect to model parameters
        loss.backward()

        # Update model parameters based on gradients
        optimizer.step()

    # Compute final loss and accuracy for the epoch
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return  # Function does not return a value



def valid(epoch, model, testing_loader, device, loss_function):
    """
    Function to validate the model's performance on the test dataset.

    Parameters:
    - epoch (int): The current epoch number.
    - model (torch.nn.Module): The trained DistilBERT model.
    - testing_loader (DataLoader): Dataloader for the validation dataset.
    - device (torch.device): The device (CPU or GPU) on which computations will be performed.
    - loss_function (torch.nn.Module): The loss function used for evaluation.

    Returns:
    - None (prints validation loss and accuracy for the epoch)
    """

    # Initialize tracking variables
    n_correct = 0  # Tracks the number of correct predictions
    tr_loss = 0  # Accumulates the total loss
    nb_tr_steps = 0  # Tracks the number of steps (batches) processed
    nb_tr_examples = 0  # Tracks the total number of examples processed

    # Set the model to evaluation mode (disables dropout and batch norm)
    model.eval()

    # Disable gradient calculations to reduce memory usage and speed up computations
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):  # Iterate over validation data batches
            # Move inputs and targets to the specified device
            ids = data['ids'].to(device, dtype=torch.long)  # Input token IDs
            mask = data['mask'].to(device, dtype=torch.long)  # Attention mask
            targets = data['targets'].to(device, dtype=torch.long)  # True labels

            # Forward pass: get model predictions
            outputs = model(ids, mask).squeeze()

            # Compute loss
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()  # Accumulate total loss

            # Get predicted class labels
            big_val, big_idx = torch.max(outputs.data, dim=1)  # Find class with highest probability
            n_correct += calculate_accu(big_idx, targets)  # Compute number of correct predictions

            # Update counters
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # Print validation loss and accuracy every 1000 steps
            if _ % 1000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation loss per 1000 steps: {loss_step}")
                print(f"Validation accuracy per 1000 steps: {accu_step}")

        # Compute overall validation loss and accuracy for the epoch
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        print(f"Validation loss at epoch {epoch} is {epoch_loss}")
        print(f"Validation accuracy at epoch {epoch} is {epoch_accu}")

    return  # No return value, results are printed

        

def main():
    """
    Main function to train and validate the DistilBERT model for text classification.
    """

    print("Start")  # Indicating the start of the training process

    # Initialize an argument parser to allow configurable parameters
    parser = argparse.ArgumentParser()

    # Define command-line arguments with default values
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimization")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access parsed arguments (though not used later in the script)
    args.epochs
    args.train_batch_size

    # Set device to GPU if available; otherwise, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the DistilBERT tokenizer from the Hugging Face model hub
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Initialize the model instance
    model = DistilBERTClass()

    # Move the model to the designated device (CPU/GPU)
    model.to(device)

    # Define the learning rate for the optimizer
    LEARNING_RATE = 1e-05

    # Initialize the Adam optimizer with the model parameters
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Define the loss function (CrossEntropyLoss for multi-class classification)
    loss_function = torch.nn.CrossEntropyLoss()

    # Define the number of epochs for training
    # 4-5 epochs are typically enough for fine-tuning, while training from scratch might require 50-100 epochs.
    EPOCHS = 2

    # Training loop for the specified number of epochs
    for epoch in range(EPOCHS):

        print(f"Starting epoch: {epoch}")  # Indicating the start of a new epoch

        # Call the train function to train the model on the training dataset
        train(epoch, model, device, training_loader, optimizer, loss_function)

        # Call the valid function to evaluate the model on the validation dataset
        valid(epoch, model, testing_loader, device, loss_function)

    # Define the directory to save the trained model (Amazon SageMaker's predefined model directory)
    output_dir = os.environ['SM_MODEL_DIR']  # SM_MODEL_DIR is a reserved environment variable in SageMaker

    # Define file paths for saving the trained model and vocabulary
    output_model_file = os.path.join(output_dir, 'pytorch_distilbert_news.bin')
    output_vocab_file = os.path.join(output_dir, 'vocab_distilbert_news.bin')

    # Save the trained model's state dictionary
    torch.save(model.state_dict(), output_model_file)

    # Save the tokenizer's vocabulary
    tokenizer.save_vocabulary(output_vocab_file)


# Ensure the main function runs when the script is executed directly
if __name__ == '__main__':   
    main()

    


        


































