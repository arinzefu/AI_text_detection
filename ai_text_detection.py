

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

data = pd.read_csv('Training_Essay_Data.csv')

data.head()

ai_text_generated = data['generated'].value_counts()

ai_text_generated

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#Tokenize the dataset
input_ids = []
attention_mask = []
for index, text in enumerate(data['text']):
    tokenized = tokenizer(text, padding="max_length", truncation=True, max_length=256)
    input_ids.append(tokenized["input_ids"])
    attention_mask.append(tokenized["attention_mask"])

input_ids = torch.tensor(input_ids, dtype=torch.long)
attention_mask = torch.tensor(attention_mask, dtype=torch.long)
labels = torch.tensor(data["generated"].values, dtype=torch.long)

print(input_ids)

print(attention_mask)

from sklearn.model_selection import train_test_split

# Split set
batch_size = 16
tokenized_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
train_data, test_data = train_test_split(tokenized_dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Create DataLoaders for training, validation, and test
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for batch_idx, batch in enumerate(train_dataloader):
    if batch_idx == 0:
        input_ids, attention_mask, labels = batch
        # Print or process the first batch here
        print("Batch 0 - Input IDs:", input_ids)
        print("Batch 0 - Attention Mask:", attention_mask)
        print("Batch 0 - Labels:", labels)
        break  # Stop after processing the first batch

from transformers import BertModel, BertTokenizer
import torch.nn as nn

class AITextDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(AITextDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

device = torch.device("cuda")

# instantiate your model
AI_text_model = AITextDetectionModel(num_classes=2).to(device)

print(AI_text_model)

# define loss function
criterion = nn.CrossEntropyLoss()

# define optimizer
optimizer = torch.optim.Adam(AI_text_model.parameters(), lr=0.01)

from tqdm import tqdm
num_epochs = 5
print_interval = 500

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    counter = 0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        # Unpack the batch into input_ids, attention_mask, and labels
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = AI_text_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # print running loss for each batch after every 500 iterations
        running_loss += loss.item()
        counter += 1

        if counter % print_interval == 0 or counter == len(train_dataloader):
            avg_loss = running_loss / counter
            avg_acc = correct_predictions / total_predictions
            tqdm.write(f'Train Loss: {avg_loss:.3f}, Train Acc: {avg_acc:.3f}', end='\r')

    # Print at the end of each epoch
    tqdm.write(f'Epoch {epoch+1}, Train Loss: {avg_loss:.3f}, Train Acc: {avg_acc:.3f}')
    print(f"Epoch {epoch+1} finished")

# Validation loop
with torch.no_grad():
    AI_text_model.eval()  # Set the model to evaluation mode
    valid_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for batch in val_dataloader:
        # Unpack the batch into input_ids, attention_mask, and labels
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        # forward
        outputs = AI_text_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # calculate running loss
        valid_loss += loss.item()

    avg_loss = valid_loss / len(val_dataloader)
    avg_acc = correct_predictions / total_predictions
    print(f'Validation Loss: {avg_loss:.3f}, Validation Acc: {avg_acc:.3f}')

# Test loop
with torch.no_grad():
    AI_text_model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for batch in test_dataloader:
        # Unpack the batch into input_ids, attention_mask, and labels
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        # forward
        outputs = AI_text_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # calculate running loss
        test_loss += loss.item()

    avg_loss = test_loss / len(test_dataloader)
    avg_acc = correct_predictions / total_predictions
    print(f'Test Loss: {avg_loss:.3f}, Test Acc: {avg_acc:.3f}')

# save the model
torch.save(AI_text_model.state_dict(), 'ai_text_model.pth')

# Sample text to evaluate
sample_text = "  Our brain is so powerful that it can easily imagine scenarios and make use of our senses. The job of such an essay is to appeal to our senses in a way that it creates an image in our minds. Hence a descriptive essay plays with at least one of our five senses (touch, smell, taste, hearing, sight)."

# List to store input IDs and attention masks
input_ids = []
attention_mask = []

# Tokenize and preprocess the sample text
tokenized = tokenizer(sample_text, padding="max_length", truncation=True, max_length=512)
input_ids.append(tokenized["input_ids"])
attention_mask.append(tokenized["attention_mask"])

# Convert input_ids and attention_mask to PyTorch Tensors
input_ids = torch.tensor(input_ids, dtype=torch.long)
attention_mask = torch.tensor(attention_mask, dtype=torch.long)

# Set the model to evaluation mode
AI_text_model.eval()

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Forward pass
with torch.no_grad():
    outputs = AI_text_model(input_ids=input_ids, attention_mask=attention_mask)
    predicted_class = torch.argmax(outputs, dim=1).item()

# Define class labels (0 for fake, 1 for real)
class_labels = ["NOT AI", "AI TEXT"]

# Get the predicted label
predicted_label = class_labels[predicted_class]

# Get the probability scores
probability_scores = torch.softmax(outputs, dim=1)
fake_probability = probability_scores[0][0].item()
real_probability = probability_scores[0][1].item()

# Print the result
print(f"Sample text: {sample_text}")
print(f"Predicted label: {predicted_label}")
print(f"Confidence - NOT AI: {fake_probability * 100:.2f}%")
print(f"Confidence - AI TEXT: {real_probability * 100:.2f}%")