# ==============================================
# Imports
# ==============================================
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm
import time
import psutil
import random
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from private_transformers import PrivacyEngine

# ==============================================
# Seed function
# ==============================================
def set_seed(seed):
    print(f"\n===== Running with seed: {seed} =====")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================
# Sanitize function
# ==============================================
nlp = spacy.load("en_core_web_sm")
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

# ==============================================
# Preprocess function
# ==============================================
def preprocess_data(dataset, tokenizer, max_length=128):
    def tokenize(batch):
        return tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=max_length)
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset

# ==============================================
# Dataloader helper
# ==============================================
def get_dataloader(dataset, batch_size=64):
    if isinstance(dataset, DatasetDict):
        dataset = dataset['train']
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ==============================================
# Train function
# ==============================================
def train(model, dataloader, optimizer, num_epochs=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader):
            inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=masks).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        accuracy = evaluate(model, dataloader)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    return avg_loss, accuracy

# ==============================================
# Evaluate function
# ==============================================
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

# ==============================================
# Main experiment function
# ==============================================
def run_experiment(seed):
    set_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load and balance training set
    datasetsst2 = load_dataset("glue", "sst2")
    train_dataset = datasetsst2["train"]

    class_0 = [ex for ex in train_dataset if ex["label"] == 0]
    class_1 = [ex for ex in train_dataset if ex["label"] == 1]
    min_size = min(len(class_0), len(class_1))
    class_0_down = random.sample(class_0, min_size)
    class_1_down = random.sample(class_1, min_size)
    balanced_data = class_0_down + class_1_down
    random.shuffle(balanced_data)
    dataset = Dataset.from_list(balanced_data)

    # Sanitize
    dataset_sanitized = dataset.map(lambda x: {'sentence': sanitize_data(x['sentence'])})

    # Tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    # Tokenize
    dataset_tokenized = preprocess_data(dataset_sanitized, tokenizer)
    dataloader_sanitized_half = get_dataloader(dataset_tokenized)

    # Step 1: Train on sanitized
    optimizer_sanitized = optim.AdamW(model.parameters(), lr=5e-5)
    print("Fine-tuning on sanitized data")
    train(model, dataloader_sanitized_half, optimizer_sanitized, num_epochs=2)
    torch.save(model.state_dict(), f"roberta_sanitized_seed{seed}.pth")

    # Validation set balancing
    v_dataset = datasetsst2["validation"]
    class_0 = [ex for ex in v_dataset if ex["label"] == 0]
    class_1 = [ex for ex in v_dataset if ex["label"] == 1]
    min_size = min(len(class_0), len(class_1))
    class_0_down = random.sample(class_0, min_size)
    class_1_down = random.sample(class_1, min_size)
    balanced_data = class_0_down + class_1_down
    random.shuffle(balanced_data)
    val_dataset = Dataset.from_list(balanced_data)
    val_dataset = val_dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=128), batched=True)
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_loader = DataLoader(val_dataset, batch_size=64)
    
    from opacus.grad_sample import GradSampleModule

    # Load sanitized model
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    # model.load_state_dict(torch.load(f"roberta_sanitized_seed{seed}.pth", weights_only=True))
    model.load_state_dict(torch.load(f"roberta_sanitized_seed{seed}.pth"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # # ✅ Wrap with GradSampleModule for DP training
    # model = GradSampleModule(model)

    # Private fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=64,
        sample_size=len(dataloader_sanitized_half.dataset),
        epochs=1,
        max_grad_norm=0.1,
        target_epsilon=3,
        clipping_mode="default"
    )
    privacy_engine.attach(optimizer)

    
#     model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
#     model.load_state_dict(torch.load(f"roberta_sanitized_seed{seed}.pth"))
#     model.to("cuda" if torch.cuda.is_available() else "cpu")

#     # Private fine-tuning
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     privacy_engine = PrivacyEngine(
#         model,
#         batch_size=64,
#         sample_size=len(dataloader_sanitized_half.dataset),
#         epochs=1,
#         max_grad_norm=0.1,
#         target_epsilon=3,
#         clipping_mode="ghost"
#     )
#     privacy_engine.attach(optimizer)

    for epoch in range(1):
        for step, batch in enumerate(dataloader_sanitized_half):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["label"].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels, reduction="none")
            optimizer.step(loss=loss)
            optimizer.zero_grad()

    # Evaluate
    accuracy = evaluate(model, test_loader)
    print(f"✅ Final Accuracy (seed {seed}): {accuracy*100:.2f}%")

# ==============================================
# Run for 10 seeds
# ==============================================
if __name__ == "__main__":
    for i in range(10):
        run_experiment(seed=42 + i)
