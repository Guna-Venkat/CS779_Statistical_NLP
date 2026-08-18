# 🧠 Character-Level Causal Language Model — Pretraining & Downstream Tasks

This project implements a **character-level causal language model (Char-LM)** from scratch using a **decoder-only Transformer architecture**, inspired by GPT-style models.
It walks through **pretraining**, **fine-tuning**, and **transfer learning**, offering both theoretical insight and practical experimentation.

---

## 🚀 Project Overview

Language models are the backbone of modern AI systems like ChatGPT and LLaMA.
While most work at the *word* or *token* level, **character-level models** learn from the smallest possible unit — the **character** — giving them finer control over spelling, grammar, and even creative language generation.

This project explores:

* Pretraining a **character-level causal LM**.
* Finetuning it for **spell-correction (denoising)**.
* Transferring learned representations to **sentiment classification**.

---

## 🧩 Why Character-Level Models?

Character-level Transformers:

* Require **no tokenizer** — just a character vocabulary.
* Handle **noisy or low-resource text** effectively (e.g., typos, code, social media).
* Learn fine-grained spelling and structure rules.
* Serve as an ideal sandbox for understanding **decoder-only Transformers** (the core of GPT).

Example generation:

```
Input : The sun se
Output: The sun sets beyond the hills.
```

Even if “sets” or “serpent” are rare words, a character model learns valid continuations **letter by letter**.

---

## ⚙️ Architecture — Decoder-Only Transformer

The model follows a **GPT-like decoder architecture**:

* **Causal Multi-Head Self-Attention:** Each character attends only to previous ones.
* **Feed-Forward Layers:** Refine intermediate representations.
* **Residual + LayerNorm:** Enable gradient flow and training stability.
* **Classification Head:** Predicts the next character.

Resources for deeper understanding:

* [Decoder-Only Transformer Diagram – Cameron Wolfe](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)
* [Hugging Face LLM Course — Transformers Illustrated](https://huggingface.co/learn/llm-course/en/chapter1/6)
* [PoloClub Interactive Transformer Visualizer](https://poloclub.github.io/transformer-explainer/)

---

## 📚 Tasks Implemented

### **1️⃣ Pretraining — Character-Level Language Modeling**

Train the LM on raw WikiText-103 data to predict the next character:

```
Input  : T h e   s u n   s e
Target : h e   s u n   s e t s
```

Metrics tracked:

* **Loss** and **Perplexity (PPL)** for generative confidence.
* Checkpointing and plot visualizations for loss trends.

---

### **2️⃣ Spell-Correction (Denoising) Fine-Tuning**

Finetune the pretrained LM to correct noisy text using examples of the form:

```
<BOS> teh quik borwn fix <SEP> the quick brown fox <EOS>
```

Here:

* `<BOS>` → Beginning of sequence
* `<SEP>` → Separator between noisy and clean text
* `<EOS>` → End of sequence

Evaluation metric:

* **Character Error Rate (CER)** — percentage of character mismatches between prediction and reference.

---

### **3️⃣ Sentiment Classification (IMDb)**

Use the pretrained character encoder for **sentence-level sentiment classification**.
Attach a lightweight classification head and fine-tune selectively.

Metrics tracked:

* Accuracy (`acc`)
* F1 Score (`f1`)
* Loss curves and ROC/AUC visualizations

---

## 📊 Training Metrics (IMDb Sentiment Task)

```json
[
  {"train_loss":0.6916,"val_loss":0.6850,"train_acc":0.5236,"val_acc":0.5528,"train_f1":0.5201,"val_f1":0.5494},
  {"train_loss":0.6773,"val_loss":0.6663,"train_acc":0.5715,"val_acc":0.5969,"train_f1":0.5624,"val_f1":0.5518},
  {"train_loss":0.6456,"val_loss":0.6296,"train_acc":0.6237,"val_acc":0.6503,"train_f1":0.6177,"val_f1":0.6825},
  {"train_loss":0.6064,"val_loss":0.5985,"train_acc":0.6714,"val_acc":0.6752,"train_f1":0.6729,"val_f1":0.7033},
  {"train_loss":0.5814,"val_loss":0.5759,"train_acc":0.6884,"val_acc":0.6948,"train_f1":0.6942,"val_f1":0.6957},
  {"train_loss":0.5641,"val_loss":0.5650,"train_acc":0.7058,"val_acc":0.7044,"train_f1":0.7107,"val_f1":0.7111},
  {"train_loss":0.5447,"val_loss":0.5495,"train_acc":0.7178,"val_acc":0.7154,"train_f1":0.7225,"val_f1":0.7261},
  {"train_loss":0.5289,"val_loss":0.5468,"train_acc":0.7320,"val_acc":0.7196,"train_f1":0.7356,"val_f1":0.7366},
  {"train_loss":0.5136,"val_loss":0.5494,"train_acc":0.7447,"val_acc":0.7249,"train_f1":0.7490,"val_f1":0.7465},
  {"train_loss":0.5022,"val_loss":0.5417,"train_acc":0.7514,"val_acc":0.7280,"train_f1":0.7550,"val_f1":0.7436}
]
```

---

## 🧩 Challenges & Observations

### **1. Data Noise and Tokenization**

**Challenges:**

* Inconsistent noise (typos, casing) → unstable mappings.
* Missing tokens or vocab drift → out-of-vocab errors.

**Observations:**

* Using `<PAD>` for unknowns stabilized training with slight CER increase.
* Clean casing helps convergence but limits robustness.
* Smaller model (n_heads=4) with tuned LR ≈ Large model (n_heads=8) performance.
  → **Smaller, well-optimized models can match larger ones.**

---

### **2. Training Stability**

**Challenges:**

* NaNs from high LR or AMP overflow.
* Embedding/vocab mismatches between pretrain & finetune.
* Layer unfreezing caused divergence.

**Observations:**

* Gradient clipping (≤1.0) + LR warmup stabilized training.
* Truncating sequences to `model.max_len` avoided embedding errors.
* Freezing encoder during sentiment fine-tune prevented catastrophic forgetting.
* **Too much unfreezing → unstable due to small data.**

---

### **3. Metric Interpretation (CER & PPL)**

**Challenges:**

* CER sensitive to spacing/punctuation.
* PPL doesn’t always match subjective fluency.

**Observations:**

* Smooth PPL & CER → healthy convergence.
* CER spikes (e.g., 18 → 84) signal gradient explosion or NaNs.
* PPL parity across models → efficiency, not size, matters.

---

### **4. Noise Injection Effect**

**Challenges:**

* Noise level tuning is critical:

  * Too little → underfitting.
  * Too much → structure loss.

**Observations:**

* Best robustness at **6–10% noise**.
* 15–20% noise → high CER, instability.
* Curriculum noise scheduling improves generalization.

---

### **5. Transfer Learning to Sentiment Task**

**Challenges:**

* Domain gap: surface-form correction → semantic polarity.
* Over-unfreezing harms small-data performance.

**Observations:**

* Freezing encoder + tuning classifier head → stable convergence.
* Pretrained encoder acts as **text normalizer**.
* Full unfreezing → oscillations & overfitting.
* Lightweight adapters outperform full retraining in low-data regimes.

---

## 🧠 Key Takeaways

✅ **Stable + Efficient Training** — Moderate noise, smaller model (n_heads=4), LR warmup, and gradient clipping.
✅ **Metrics Matter** — Track CER for stability, PPL for confidence.
✅ **Effective Transfer Learning** — Freeze base layers, tune only task-specific heads.
✅ **Smaller Models ≈ Large Models** — When hyperparameters and data prep are optimized.

---

## 📁 Repository Structure

```
CS779-A5-Guna-Venkat-Doddi-251140009/
├─ notebooks/
│  └─ CS779-A5-Guna-Venkat-Doddi-251140009.ipynb
│
├─ experiments/
│  ├─ results/
│  │  ├─ q1/
│  │  │  ├─ meta.json
│  │  │  └─ log_pretrain.csv              
│  │  ├─ q2/
│  │  │  └─ log_spell_f1.csv                   
│  │  │   
│  │  └─ q3d/
│  │     ├─ metrics.csv       
│  │     ├─ history.json             
│  │     └─ roc_points.csv                
│  └─ figures/
│     ├─ q1/
│     │  ├─ plot_pretrain_train_loss.png
│     │  ├─ plot_pretrain_val_loss.png
│     │  └─ plot_pretrain_val_ppl.png
│     ├─ q2/
│     │  ├─ plot_train_loss.png          
│     │  └─ plot_val_cer.png                 
│     └─ q3d/
│        ├─ acc_curve.png                 
│        ├─ loss_curve.png                
│        └─ roc_curve.png                 
│
└─ README.md```

---

## 🧮 References

* Vaswani et al., *Attention Is All You Need* (2017)
* Andrej Karpathy, *nanoGPT*
* Hugging Face, *LLM Course*
* Cameron R. Wolfe, *Decoder-Only Transformers Explained*
* Poloclub, *Transformer Explainer*

---

## ✨ Author

**Guna Venkat Doddi**

---