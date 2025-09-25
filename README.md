# Differentially-Private Prompting in Large Language Models (DualTune-GhostDP)

## Overview
With growing concerns about data privacy and confidentiality, privacy-preserving techniques are becoming essential for data-driven applications, especially **Large Language Models (LLMs)**. LLMs excel at in-context learning and are widely used in real-world applications, but they often rely on sensitive private data, making them vulnerable to **data leakage** and **privacy breaches**.

Our project proposes **DualTune-GhostDP**, a **two-phase fine-tuning framework** for LLMs that balances **privacy and utility**. By using **Ghost Clipping** and controlling privacy with the **EW Advanced Accountant** instead of traditional privacy accountants, our model maintains strong Differential Privacy (DP) guarantees while achieving high performance and improved computational efficiency.

## Features
- Two-phase fine-tuning for LLMs  
- Differential Privacy via Ghost Clipping  
- Privacy accounting using EW Advanced Accountant  
- Faster and more memory-efficient training  
- Improved accuracy compared to:  
  - Single-phase DP-SGD  
  - Two-phase fine-tuning with traditional clipping  

## Installation
1. Clone this repository:  
```bash
git clone https://github.com/AnonymousSatMLll/Differentially-Private-Prompting-in-Large-Language-Models.git```

2. Install required libraries:
```bash
pip install -r requirements.txt

3. Run any experiment for a dataset using:
```bash
python [nameofexperiment].py
Replace [nameofexperiment] with the script name for the experiment you want to run.
