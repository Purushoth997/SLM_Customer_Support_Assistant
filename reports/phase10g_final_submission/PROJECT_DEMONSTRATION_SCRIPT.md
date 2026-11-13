# Mentor Demonstration Script: SLM Customer Support Assistant

**Demonstration Duration:** 2.5 to 3 Minutes  
**Target Audience:** Project Mentor / Technical Evaluator  
**Application URL:** `http://127.0.0.1:8000/`  
**Execution Environment:** VS Code on Windows 11 CPU (Intel Core i5-12450H, 16 GB RAM)  

---

## 1. Demonstration Overview & Timing Guide

| Phase | Section Title | Target Time | Key Talking Points |
| :---: | :--- | :---: | :--- |
| **01** | **Introduction & Problem Statement** | 0:00 – 0:30 | Enterprise privacy, token economics, why Qwen2.5-1.5B SLM. |
| **02** | **Application Launch & Architecture** | 0:30 – 1:00 | VS Code environment, FastAPI backend, dynamic PEFT loading. |
| **03** | **Live Interactive Inquiries** | 1:00 – 2:00 | Return policy, order tracking, and multi-turn damaged item support. |
| **04** | **Explaining Dynamic LoRA Inference** | 2:00 – 2:30 | Why weights remain unmerged ($W = W_0 + \Delta W$), PEFT efficiency. |
| **05** | **Results, Disclosed Nuances & Conclusion** | 2:30 – 3:00 | +17.2% F1 gain, CPU trade-offs, transparent limitations disclosure. |

---

## 2. Step-by-Step Walkthrough

### Part 1: Project Introduction (0:00 – 0:30)
> *"Hello! Today I am presenting our **SLM-Powered Customer Support Assistant**. 
> 
> Modern enterprise customer service often relies on frontier cloud LLMs like GPT-4, which introduce recurring per-token costs, high latency variances, and third-party data privacy exposure. 
> 
> In this project, we demonstrate that a domain-adapted Small Language Model—specifically **Qwen2.5-1.5B-Instruct** fine-tuned with **QLoRA**—can handle enterprise customer inquiries locally. It achieves 100% data privacy, predictable infrastructure costs, a 55.4% reduction in wordy filler, and a +17.2% improvement in reference token overlap, all running right here on standard Windows CPU hardware."*

---

### Part 2: Starting the Application in VS Code (0:30 – 1:00)
> *"Let's look at how the application is structured and launched.
> 
> Here in VS Code, we have an isolated virtual environment, `venv-peft`, containing CPU-optimized PyTorch 2.5.1, Hugging Face Transformers, and PEFT.
> 
> To launch the application, we start our FastAPI backend service:
> ```powershell
> # In VS Code Terminal
> .\venv-peft\Scripts\activate
> python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
> ```
> The server initializes the base model from our local cache and dynamically binds our fine-tuned LoRA adapter (`customer-support-lora`).
> 
> Now, opening our browser to `http://127.0.0.1:8000/`, we see our single-page customer support console. You can see the real-time status indicator showing **'Online (LoRA Active)'** and our system telemetry reporting active CPU execution."*

---

### Part 3: Live Interactive Queries (1:00 – 2:00)

#### Demonstration Query 1: Policy Grounding (Return Policy)
- **Action:** In the chat input, submit:
  > *"What is your return policy for open-box electronics?"*
- **Supplied Context (or paste into Context box if using advanced mode):**
  > *"Electronics can be returned within 14 days of delivery if in original box with all accessories. A 15% restocking fee applies to open-box items unless defective."*
- **Talking Point while generation runs (~12–14 seconds on CPU):**
  > *"Notice that the model generates its response on CPU. The response is concise and disciplined: it accurately identifies the 14-day window and the 15% restocking fee, avoiding generic conversational preamble."*

#### Demonstration Query 2: Order Tracking Inquiry
- **Action:** Submit:
  > *"Where is my order? Order ID is APX-88341."*
- **Supplied Context:**
  > *"Order APX-88341 was shipped via UPS ground on Nov 11, tracking number 1Z999AA10123456784, currently out for delivery."*
- **Talking Point:**
  > *"Here the model extracts the carrier—UPS Ground—and confirms the package is in transit. It adheres to customer service tone and empathy without fabricating unrealistic delivery guarantees."*

#### Demonstration Query 3: Multi-Turn Conversation (Damaged Goods & Courier)
- **Action Turn 1:** Submit:
  > *"My delivery arrived today but the glass vase inside was completely shattered."*
- **Action Turn 2 (Follow-up):** Submit:
  > *"Can FedEx hold my replacement at a pickup location instead?"*
- **Talking Point:**
  > *"In this multi-turn dialogue, the assistant maintains conversational context across turns. It empathetically apologizes, instructs the customer to submit photos for a free replacement, and advises them on utilizing carrier pickup features."*

---

### Part 4: Explaining Dynamic LoRA Inference (2:00 – 2:30)
> *"A key architectural strength of this application is **Dynamic PEFT Loading**. 
> 
> If we inspect `http://127.0.0.1:8000/model/info`:
> ```json
> {
>   "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
>   "adapter_name": "customer-support-lora",
>   "lora_rank": 8,
>   "lora_alpha": 16,
>   "dynamic_peft": true,
>   "merged_weights": false
> }
> ```
> Notice that `dynamic_peft` is `true` and `merged_weights` is `false`. 
> 
> Rather than merging the 9.2 million adapter weights into the 1.54 billion base model weights permanently, Hugging Face PEFT dynamically computes the forward pass as:
> $$W_{\text{effective}} \cdot x = W_0 \cdot x + \frac{\alpha}{r} (B \cdot A \cdot x)$$
> This design means our base model snapshot remains completely immutable and reusable, while multiple domain adapters (e.g., technical support, billing, or sales) can be dynamically swapped in memory without storing duplicate 3 GB base model checkpoints."*

---

### Part 5: Results & Disclosed Nuances (2:30 – 3:00)
> *"Finally, let us review our evaluation results and observed nuances:
> 
> 1. **Evaluation Gains:** In our official holdout evaluation on 50 quarantined test records, the fine-tuned adapter achieved a **+17.2% improvement in Mean Token F1** (0.2891 vs 0.2467) and reduced average completion length from 78.7 tokens down to 35.1 tokens.
> 2. **CPU Inference Trade-off:** On our Intel i5 CPU, generation takes ~12 to 19 seconds per answer (~2.5 tokens/sec). While slower than cloud GPUs (which ran at 3.8s), this enables zero-cost local prototyping on standard office laptops.
> 3. **Transparent Limitations:** As rigorous engineers, we disclose that in live tests:
>    - The model once described a 14-day policy as starting from *purchase* date rather than *delivery* date.
>    - It truncated a long tracking string.
>    - Its carrier guidance is advisory based on policy, not an active carrier API call.
>    - Therefore, in production, we recommend pairing the SLM with an upstream deterministic regex router for high-liability inquiries.
> 
> Thank you, and I look forward to your questions!"*

---

## 3. Mentor Q&A Cheat Sheet

| Potential Mentor Question | Recommended Technical Answer |
| :--- | :--- |
| **Q1: Why didn't you merge the LoRA weights into the base model?** | *"Keeping weights unmerged enables modular multi-adapter serving. One base model instance (3.1 GB) can serve multiple distinct domain adapters (35 MB each) dynamically, rather than storing multiple 3 GB merged models on disk. It also preserves base model immutability."* |
| **Q2: Why use Qwen2.5-1.5B instead of Llama-3.2-1B or Phi-3.5?** | *"Qwen2.5-1.5B has native ChatML template support, superior multi-lingual and structured reasoning benchmarks, and Grouped Query Attention (GQA) with 12 query and 2 KV heads, which significantly optimizes memory bandwidth during inference."* |
| **Q3: How do you address the ~15-second CPU latency in production?** | *"For high-volume production, two deployment paths exist: (1) Serving on a cloud GPU (e.g., Tesla T4 / L4), which drops latency to 3.8s at 9.3 tok/s, or (2) Exporting the model to 4-bit GGUF via llama.cpp, which achieves ~3–4s latency on modern CPUs with AVX2 acceleration."* |
| **Q4: How do you ensure the model doesn't hallucinate refund approvals?** | *"Our evaluation proved that generative models should not be the sole authority for financial commitments. We recommend an upstream deterministic router for refunds and legal threats, combined with downstream regex screening for financial commitments."* |
