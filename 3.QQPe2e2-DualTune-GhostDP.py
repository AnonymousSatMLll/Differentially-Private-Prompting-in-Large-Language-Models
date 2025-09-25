# Update datasets and fsspec libraries
import spacy
###Clear memory
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset, DatasetDict, Dataset
# from opacus import PrivacyEngine
# from opacus.grad_sample import GradSampleModule
# from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import time
import psutil
import random
from transformers import pipeline
# =========================
# Global SEED setup
# =========================
import numpy as np

SEED = 54  # <-- Try 42+i (i=1 to 10)
print(f"\n===== Running with seed: {SEED} =====\n")

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
nlp = spacy.load("en_core_web_sm")

# Load QQP dataset
datasetqqp = load_dataset("glue", "qqp")

import random
from datasets import Dataset

train_dataset = datasetqqp["train"]
label_counts = {0: 0, 1: 0}
for example in train_dataset:
    label_counts[example["label"]] += 1
print(f"Original class counts: {label_counts}")

# Separate by class
class_0 = [ex for ex in train_dataset if ex["label"] == 0]
class_1 = [ex for ex in train_dataset if ex["label"] == 1]

# Balance by downsampling to minority class size
min_size = min(len(class_0), len(class_1))
random.seed(42)
class_0_down = random.sample(class_0, min_size)
class_1_down = random.sample(class_1, min_size)

# Combine and shuffle
balanced_data = class_0_down + class_1_down
random.shuffle(balanced_data)

# Convert to Huggingface Dataset
dataset = Dataset.from_list(balanced_data)
balanced_dataset = dataset
# Verify class counts
new_counts = {0: 0, 1: 0}
for ex in dataset:
    new_counts[ex["label"]] += 1
print(f"Balanced class counts: {new_counts}")



# Display the type of the dataset
print(type(dataset))
#High contextual Secret Detector
def sanitize_data(text):
    doc = nlp(text)
    redacted = []
    for token in doc:
        if (token.ent_type_ or token.pos_ in {"PRON", "PROPN", "VERB"}
            or token.dep_ in {"nsubj", "dobj", "pobj"}):
            redacted.append(f"<{token.pos_}>")
        else:
            redacted.append(token.text)
    return ' '.join(redacted)


# Preprocess data
def preprocess_data(dataset, tokenizer, max_length=128):
    def tokenize(example):
        return tokenizer(example["question1"], example["question2"],
                     padding="max_length", truncation=True, max_length=max_length, return_overflowing_tokens=False)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset
####################################### Step 1: Sanitize and preprocess dataset
# Apply redaction
print("Sanitizing Data:\n")
dataset_sanitized = dataset.map(
    lambda x: {'question1': sanitize_data(x['question1']), 'question2': sanitize_data(x['question2'])}
  )


print("\nHigh Contextual Redacted Data\n")
# Iterate over first 5 examples
for i in range(5):
    print(f"{i+1}. Question 1: {dataset_sanitized[i]['question1']}")
    print(f"   Question 2: {dataset_sanitized[i]['question2']}\n")
    


# Load Tokenizer and pre-trained model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# MODEL_NAME = 'distilbert-base-uncased'
# tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
# model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Apply tokenization
dataset_tokenized = preprocess_data(dataset_sanitized, tokenizer)

# Get dataloader for sanitized data (50% of the dataset for quicker training)
def get_dataloader(dataset, batch_size=64):
    # Check if dataset is a DatasetDict and access the 'train' split if necessary
    if isinstance(dataset, DatasetDict):
        dataset = dataset['train']
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Dataloader for sanitized (half) dataset
dataloader_sanitized_half = get_dataloader(dataset_tokenized)



################# Training function
# Define training function
def train(model, dataloader, optimizer, dp_enabled=False, use_ghost=False, noise_multiplier=1.0, max_grad_norm=1.0, num_epochs=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    privacy_engine = None
    if dp_enabled:
        model = GradSampleModule(model)
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            clipping='ghost'
        )

    for epoch in range(num_epochs): # Changed to use num_epochs parameter
        running_loss = 0.0
        for batch in tqdm(dataloader):
            inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=masks).logits
            loss = criterion(outputs, labels)
            loss.backward()

            if use_ghost:
                for layer in model.children():
                    clip_grad_norm_(layer.parameters(), max_grad_norm)  # Clip for each layer

            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)

        # Compute and print Accuracy at each epoch
        accuracy = evaluate(model, dataloader) # Calculate accuracy using evaluate function
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    if dp_enabled:
        epsilon, best_alpha = privacy_engine.get_privacy_spent(1e-5)
        print(f"(ε, δ) = ({epsilon}, 1e-5) for best α={best_alpha}")

    return avg_loss, accuracy # Return both avg_loss and accuracy

################# Evaluate function
def evaluate(model, dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            outputs = model(input_ids=inputs, attention_mask=masks).logits
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy

################################################################################
# Step 1: Fine-tune on sanitized data
optimizer_sanitized = optim.AdamW(model.parameters(), lr=5e-5)
print("Fine-tuning on sanitized data")
loss_sanitized, acc_sanitized = train(model, dataloader_sanitized_half, optimizer_sanitized, dp_enabled=False, num_epochs=2)
torch.save(model.state_dict(), "roberta_sanitized.pth")

# torch.save(model.state_dict(), "bert_sanitized.pth")

# Evaluate after first fine-tuning
evaluate(model, dataloader_sanitized_half)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from private_transformers import PrivacyEngine
import matplotlib.pyplot as plt
import os
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Display the type of the dataset
dataset= balanced_dataset
print(type(dataset))
# Optional: Print final class distribution to verify
from collections import Counter
print(Counter([ex['label'] for ex in dataset]))

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Tokenize
def tokenize(example):
    return tokenizer(example["question1"], example["question2"],
                     padding="max_length", truncation=True, max_length=128, return_overflowing_tokens=False)


dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# # # Take 50000 samples from train split
# # subset_indices = list(range(50000))
# # train_dataset = Subset(dataset, subset_indices)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Balance validation set
# Load the validation set from the original dataset
dataset = load_dataset("glue", "qqp")
v_dataset = dataset["validation"]

# Step 1: Separate examples by label
class_0 = [ex for ex in v_dataset if ex["label"] == 0]
class_1 = [ex for ex in v_dataset if ex["label"] == 1]
###Step 2: Downsample both classes to 5,000 each
# Balance by downsampling to minority class size
min_size = min(len(class_0), len(class_1))
random.seed(42)
class_0_down = random.sample(class_0, min_size)
class_1_down = random.sample(class_1, min_size)
# Step 3: Combine and shuffle the balanced subset
balanced_data = class_0_down + class_1_down
random.shuffle(balanced_data)

from collections import Counter
print(Counter([ex['label'] for ex in balanced_data]))

# Step 4: Convert to HuggingFace Dataset
val_dataset = Dataset.from_list(balanced_data)
#Tokenized val_DS
val_dataset = val_dataset.map(tokenize, batched=True)
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
# Test split
test_loader = DataLoader(val_dataset, batch_size=64)


# Define the model architecture again
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)



# Step 2: Load sanitized model for private fine-tuning
model.load_state_dict(torch.load("roberta_sanitized.pth"))

###Memory usage
torch.cuda.reset_peak_memory_stats()
model.train()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
print(f"Private fine-tuning training size: {len(train_loader.dataset)}")

# Privacy Engine
privacy_engine = PrivacyEngine(
    model,
    batch_size=64,
    sample_size=len(train_loader.dataset),
    epochs=2,
    max_grad_norm=0.1,
    target_epsilon=3,
    clipping_mode="ghost",  # ghost clipping
)
privacy_engine.attach(optimizer)

# Trackers
train_losses = []
epsilons = []

# Training loop
for epoch in range(2):
    epoch_loss = 0.0
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels, reduction="none")

        optimizer.step(loss=loss)
        optimizer.zero_grad()

        epoch_loss += loss.mean().item()

        if step % 100 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.mean().item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Get the epsilon after each epoch
    privacy_spent = privacy_engine.get_privacy_spent(steps=epoch + 1)  # Get privacy spent so far

    # Check the structure of the returned value
    if isinstance(privacy_spent, tuple):  # If it's a tuple, the first element should be epsilon
        eps = privacy_spent[0]
    elif isinstance(privacy_spent, dict):  # If it's a dictionary, look for the key 'epsilon'
        eps = privacy_spent.get('epsilon', None)
    else:
        eps = None

    epsilons.append(eps)

    if eps is not None:
        print(f">>> Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, ε: {eps:.2f}")
    else:
        print(f">>> Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, ε: Not Available")

        
##
torch.cuda.synchronize()
max_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
print(f"Peak memory usage: {max_mem:.2f} MB")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
final_epsilon = epsilons[-1]
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

###Memory usage

print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
