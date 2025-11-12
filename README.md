# 🤖 Prompt Optimizer (Chain of Thought) – Intelligent Prompt Tuning Engine 🧠⚙️

---

## 🚀 **Overview**
An **AI-driven prompt optimization framework** that dynamically learns the **best phrasing and temperature combination** for LLMs using a **UCB1 reinforcement learning algorithm**. The optimizer enhances **Chain-of-Thought (CoT)** reasoning by testing and evaluating multiple **prompt variants** across creativity levels, ultimately discovering the most accurate, coherent, and cost-efficient configuration

This project integrates **reinforcement learning, prompt evaluation, and automated tuning**, allowing users to **optimize prompts intelligently** for reasoning tasks, Q&A, or mathematical problem solving

---

## 🔑 **Key Features:**

### 🧠 **Automated Prompt Optimization**
- Implements the **UCB1 Multi-Armed Bandit** algorithm to balance exploration and exploitation
- Automatically learns which **(phrasing × temperature)** setup yields the best model performance

### 💬 **Adaptive Chain-of-Thought Prompts**
- Fixed reasoning style: *“Think step by step; output only the final answer”*  
- Includes multiple phrasing variations to test different reasoning paths:  
  - **plain** → base CoT prompt  
  - **units_hint** → adds unit-specific clarity  
  - **bullet_pref** → organizes internal reasoning steps  
  - **math_hint** → encourages careful computation for numerical tasks  

### 🧩 **Evaluation & Scoring**
- Combines multiple performance metrics into one unified score:  
  - **Accuracy (fuzzy match)** using RapidFuzz  
  - **Coherence (clarity of expression)** based on token diversity  
  - **Cost (efficiency)** using token usage as a lightweight penalty  
- Produces a weighted score:  
  `Score = 0.9 × Accuracy + 0.1 × Coherence - 0.01 × Cost`

### 🤖 **Pluggable Model Clients**
- **DummyClient (Offline Mode):** Simulates model responses locally without API usage 
- **OpenAIClient (Live Mode):** Connects to OpenAI’s GPT models (e.g., *gpt-4o-mini*) for real testing

---

## 🧩 **Architecture**
```python
OptiPrompt/
├─ __main__.py          # Main entry and training loop
├─ config.py            # Global flags, weights, and model settings
├─ types.py             # Data structures (Params, Arm, Trial)
├─ bandits.py           # UCB1 multi-armed bandit logic
├─ evals.py             # Fuzzy accuracy and coherence metrics
├─ prompts.py           # CoT base template and phrasing variants
├─ optimizer.py         # Core optimization engine
├─ models.py            # Model adapters (Dummy + OpenAI)
└─ utils.py             # Helper functions (_truncate, _rule)
```

---

## 🧠 **How It Works**
1. **Build Arms** – Generates all (phrasing × temperature) combinations 
2. **Run Trials** – Tests each arm on a dataset using the UCB1 selection rule 
3. **Score Responses** – Evaluates accuracy, coherence, and token efficiency 
4. **Reinforce Learning** – Updates bandit rewards to prioritize top-performing arms  
5. **Output Insights** – Displays the best prompt setup per input and overall best configuration  

---

## 🛠️ **Technologies Used**
- **Programming Language:** Python 3.10+  
- **Core Algorithms:** UCB1 Multi-Armed Bandit (Reinforcement Learning)  
- **Libraries:** RapidFuzz, OpenAI API  
- **Evaluation Metrics:** Fuzzy Accuracy, Coherence, Cost Efficiency  

---

## 🎯 **Impact**
- Enables **data-driven prompt engineering** instead of manual trial and error 
- Automatically learns **optimal reasoning phrasing and temperature**  
- Improves **accuracy**, **clarity**, and **efficiency** across model responses
- Creates a **reproducible, research-ready** framework for LLM optimization

---

## 🧪 **Example Dataset**
```python
DATA = [
    {"input": "Who wrote the Harry Potter Books?", "reference": "J.K. Rowling"},
    {"input": "Compute 13 + 31", "reference": "44"},
    {"input": "Capital of France?", "reference": "Paris"},
    {"input": "Derivative of x^3 + 3x^2?", "reference": "3x^2 + 6x"},
]
```

## **Installation**
# Clone the repository (Bash)
git clone https://github.com/AkashLak/OptiPrompt.git

cd OptiPrompt
# Install dependencies
pip install rapidfuzz openai

Optional: export OPENAI_API_KEY="your_api_key_here"
# Run the program
python -m OptiPrompt

## **Behaviour**
Offline Mode: Uses DummyClient for simulated responses (no API key needed)

Live Mode: Uses OpenAIClient for real GPT outputs (requires API key)
