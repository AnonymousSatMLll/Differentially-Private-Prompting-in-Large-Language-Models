

###bert|QNLI|Normal DP-SGD
# !pip install transformers datasets
import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset, Dataset
from private_transformers import PrivacyEngine
import matplotlib.pyplot as plt

# =========================
# Global SEED setup
# =========================
import random
import numpy as np
import torch

SEED = 42  # <-- Try 42+i (i=1 to 10)
print(f"\n===== Running with seed: {SEED} =====\n")

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
    
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load QNLI dataset
dataset = load_dataset("glue", "qnli")
train_dataset = dataset["train"]

####Balancing DS
# Count class distribution
label_counts = {0: 0, 1: 0}
for example in train_dataset:
    label_counts[example["label"]] += 1

print(f"Original class counts: {label_counts}")

# Separate classes
class_0 = [ex for ex in train_dataset if ex["label"] == 0]  # Not duplicate (majority)
class_1 = [ex for ex in train_dataset if ex["label"] == 1]  # Duplicate (minority)

# Downsample majority class to match minority class
min_size = min(len(class_0), len(class_1))
random.seed(42)
class_0_down = random.sample(class_0, min_size)
class_1_down = random.sample(class_1, min_size)  # optional, to shuffle

# Combine and shuffle all DS
balanced_data = class_0_down + class_1_down
random.shuffle(balanced_data)

# Convert to DatasetDict format
balanced_dataset = Dataset.from_list(balanced_data)
print(f"Balanced dataset size: {len(balanced_dataset)}")

# Count new class distribution
new_label_counts = {0: 0, 1: 0}
for example in balanced_data:
    new_label_counts[example["label"]] += 1

print(f"Balanced class counts after subsetting: {new_label_counts}")

###Tokenization
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["question"], example["sentence"],
                     padding="max_length", truncation=True, max_length=128, return_overflowing_tokens=False)




# Tokenize and format
balanced_dataset = balanced_dataset.map(tokenize, batched=True)
balanced_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Dataloader
train_loader = DataLoader(balanced_dataset, batch_size=32, shuffle=True)



# Balance validation set
# Load the validation set from the original dataset
dataset = load_dataset("glue", "qnli")
val_dataset = dataset["validation"]

# Balance validation set
val_class_0 = [ex for ex in val_dataset if ex["label"] == 0]
val_class_1 = [ex for ex in val_dataset if ex["label"] == 1]
val_min_size = min(len(val_class_0), len(val_class_1))

random.seed(42)
val_class_0_down = random.sample(val_class_0, val_min_size)
val_class_1_down = random.sample(val_class_1, val_min_size)
val_balanced_data = val_class_0_down + val_class_1_down
random.shuffle(val_balanced_data)


# Count new class distribution
new_label_counts = {0: 0, 1: 0}
for example in val_balanced_data:
    new_label_counts[example["label"]] += 1

print(f"Balanced class counts for validation: {new_label_counts}")

# Convert to HuggingFace Dataset object
from datasets import Dataset
val_dataset = Dataset.from_list(val_balanced_data)

# Tokenize and format
val_dataset = val_dataset.map(tokenize, batched=True)
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Validation  test split DataLoader
test_loader = DataLoader(val_dataset, batch_size=64)

# # Test split
# test_loader = DataLoader(dataset["validation"], batch_size=32)

# Model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
model.train()

print("\n\nOne-phase DPSGD for QNLI\n\n")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


print(f"Private fine-tuning training size: {len(train_loader.dataset)}")
# Privacy Engine
privacy_engine = PrivacyEngine(
    model,
    batch_size=64,
    sample_size = len(train_loader.dataset),
    epochs=4,
    max_grad_norm=0.1,
    target_epsilon=3,
     clipping='default'   #Normal
)
privacy_engine.attach(optimizer)

# Trackers
train_losses = []
epsilons = []

# Training loop
for epoch in range(4):
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
print(f"\n Test Accuracy after 2 epochs for one-phase normal clipping on QNLI DS is : {accuracy * 100:.2f}%")

# # Plot
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')

# plt.title("Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")

# plt.subplot(1, 2, 2)
# plt.plot(range(1, 4), epsilons, marker='o', color='green')
# plt.title("Privacy Budget (ε) vs Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Epsilon (ε)")

# plt.tight_layout()
# plt.show()