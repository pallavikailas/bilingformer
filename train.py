import sys
sys.path.append('src')  # Include the src directory in the Python path

from src.preprocess import load_and_preprocess_data
from src.dataset import CustomDataset
from src.model import initialize_model, train_and_evaluate, save_model
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch

# File path
data_file_path = './data/MaSaC_train_efr.json'

# Load and preprocess data
train_df, val_df = load_and_preprocess_data(data_file_path)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
max_length = 64  # Adjust as needed

# Prepare datasets and dataloaders
train_dataset = CustomDataset(train_df, tokenizer, max_length)
val_dataset = CustomDataset(val_df, tokenizer, max_length)
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=32)

# Initialize model, optimizer, and criterion
num_labels = len(train_df['labels'].unique())
model, optimizer, criterion = initialize_model(num_labels)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train and evaluate the model
train_and_evaluate(model, train_dl, val_dl, device, optimizer, criterion, num_epochs=3)

# Save the model
model_save_path = './src/bert_classification_model'
save_model(model, model_save_path)
