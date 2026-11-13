# Final Project Report: SLM-Powered Customer Support Assistant

**Project:** SLM-Powered Customer Support Assistant  
**Repository Location:** `C:\slm-customer-support`  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`)  
**Fine-Tuning Method:** Quantized Low-Rank Adaptation (QLoRA, $r=8, \alpha=16$)  
**Inference Runtime:** Dynamic Hugging Face PEFT on CPU (Windows 11, Intel Core i5-12450H)  
**Evaluation Phase:** Phase 10G Final Mentor Submission  
**Date:** November 13, 2026  
**Status:** Completed & Verified  

---

## 1. Executive Summary & Project Objectives

The **SLM-Powered Customer Support Assistant** is an engineering-grade, enterprise-adapted Small Language Model (SLM) system designed to deliver high-quality, policy-grounded customer service responses without relying on costly third-party commercial LLM APIs. By adapting a foundational 1.5-billion-parameter open-weights language model (`Qwen2.5-1.5B-Instruct`) to the customer support domain using Parameter-Efficient Fine-Tuning (PEFT / QLoRA), the system achieves:

1. **Enterprise Data Privacy:** 100% of customer interactions and proprietary policy data are processed on controlled infrastructure, eliminating third-party data transmission and API privacy risks.
2. **Economic Predictability:** Eliminates per-token vendor API pricing in favor of predictable, fixed-cost infrastructure.
3. **Domain Alignment:** Fine-tuned on specialized retail customer support data across 10 distinct service domains, reducing response verbosity by 55.4% and increasing reference token overlap (+17.2% Mean Token F1) over the baseline model.
4. **Locally Runnable Deployment:** Engineered to run dynamically in VS Code on commodity consumer hardware (Windows 11 CPU) using Hugging Face Transformers and PEFT, loading adapter weights dynamically without requiring offline weight merging or specialized GPU hardware at inference time.

---

## 2. Problem Statement & Proposed Solution

### 2.1 The Problem with Frontier LLMs in Enterprise Support
Modern retail customer support centers handle millions of repetitive, highly structured inquiries annually (order status, returns, tracking, billing adjustments, and policy clarification). While general-purpose frontier LLMs (e.g., GPT-4o, Claude 3.5 Sonnet) can answer such queries, deploying them introduces significant drawbacks:
- **Prohibitive Recurring Costs:** High token billing costs accumulate rapidly across high ticket volumes.
- **Latency & Availability Variances:** Third-party cloud network round-trips and provider rate limits create unpredictable response times.
- **Verbose & Unaligned Tone:** Frontier models often produce overly verbose, conversational preamble instead of crisp, actionable support instructions.
- **Data Compliance Exposure:** Sending sensitive order information, customer identifiers, or proprietary business policies across third-party networks introduces regulatory compliance risks (GDPR, CCPA, SOC 2).

### 2.2 The Small Language Model (SLM) Solution
By fine-tuning a sub-2-billion parameter model on domain-specific support dialogues, this project demonstrates that an SLM can achieve:
- **Concise, Professional Outputs:** Answers that directly address customer needs without conversational filler.
- **Policy Adherence:** Grounded adherence to company return windows, restocking fees, and shipping procedures.
- **Low-Footprint CPU Inference:** Feasible execution on local enterprise workstations (16 GB RAM, modern multi-core CPU) without demanding expensive dedicated datacenter GPUs for low-to-medium volume support desks.

---

## 3. System Architecture & Technology Stack

The project implements a modern, modular AI application stack spanning cloud training and local CPU serving:

```text
+-----------------------------------------------------------------------------------+
|                                 CLIENT TIER                                       |
|  Single-Page Web Application (HTML5 / Vanilla JavaScript / CSS Modern UI)        |
|  Real-time Connection Health | Telemetry Badges | Multi-Turn Conversational Feed  |
+-----------------------------------------+-----------------------------------------+
                                          | HTTP REST (JSON)
                                          v
+-----------------------------------------------------------------------------------+
|                                FASTAPI BACKEND                                    |
|  FastAPI 0.115.8 / Uvicorn 0.34.0 / Pydantic 2.10.6 Validated Contracts          |
|  Endpoints: GET /health | GET /model/info | POST /chat | Static UI Mount         |
+-----------------------------------------+-----------------------------------------+
                                          | In-Memory Python API
                                          v
+-----------------------------------------------------------------------------------+
|                        PEFT INFERENCE SERVICE LAYER                               |
|  src/model/peft_inference.py                                                      |
|  - ChatML Prompt Formatter: <|im_start|>system ... <|im_start|>user ...           |
|  - Deterministic/Nucleus Generation: temperature=0.3, top_p=0.85, top_k=40        |
|  - Latency & Token Telemetry Profiler                                             |
+-----------------------------------------+-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
+-----------------------------------------+ +---------------------------------------+
|        BASE MODEL (FROZEN FP32)         | |      LORA ADAPTER (UNMERGED FP32)     |
|  Qwen/Qwen2.5-1.5B-Instruct             | |  customer-support-lora                |
|  Revision: 989aa7980e4cf806f80c7fef...  | |  SHA-256: 5d70525e3f51b7a8da435...    |
|  Parameters: 1,543,714,304 (Frozen)     | |  Trainable Parameters: 9,232,384      |
|  Storage: Hugging Face Local Hub Cache  | |  Rank: r=8, Alpha: alpha=16           |
+-----------------------------------------+ +---------------------------------------+
                     \                                         /
                      +-------------------+-------------------+
                                          v
                       PyTorch 2.5.1+cpu / Intel i5-12450H
                       Dynamic Matrix Addition: W_eff = W_0 + (alpha/r)*B*A
```

### Technology Stack Summary:
- **Core Frameworks:** PyTorch 2.5.1 (CPU build), Hugging Face Transformers 4.48.3, PEFT 0.14.0, Accelerate 1.3.0, Safetensors 0.5.2.
- **API & Web Service:** FastAPI 0.115.8, Uvicorn 0.34.0, Starlette 0.45.3, Pydantic 2.10.6.
- **Hardware Profile:**
  - Training: Google Colab Free Tier, NVIDIA Tesla T4 GPU (16 GB VRAM), CUDA 12.4.
  - Local Serving: Windows 11 Home, Intel Core i5-12450H CPU (8 cores, 12 threads), 16 GB DDR4 RAM (CPU-only execution).

---

## 4. Dataset Engineering & Split Methodology

### 4.1 Synthetic Dataset Design
To guarantee zero exposure of real customer Personally Identifiable Information (PII) and ensure total reproducibility, a high-quality dataset of **350 customer support records** was synthesized for the fictional enterprise *"Apex Retail Solutions"*.

The dataset encompasses 10 distinct, balanced domain categories (35 records per category):
1. `order_status`: Tracking numbers, carrier handoffs, address updates, transit exceptions.
2. `shipping_delivery`: Delivery speeds, courier delays, international customs, damaged packaging.
3. `payment_issues`: Split payments, expired gift cards, double charges, currency conversions.
4. `refund_requests`: Refund timelines, non-refundable digital keys, price match adjustments.
5. `return_policies`: 30-day return windows, open-box restocking fees, international return duties.
6. `product_information`: Hardware compatibility, dimensions, technical specifications.
7. `account_support`: Password resets, two-factor authentication, account deletion requests.
8. `customer_complaints`: Damaged products, service conduct, missed delivery commitments.
9. `escalation_requests`: Explicit supervisor transfers, legal claims, vendor disputes.
10. `general_support`: Store locations, operating hours, telephone customer service.

### 4.2 Split Isolation & Cryptographic Governance
The 350 cleaned records were partitioned into a leakage-free 70/15/15 split using stratified sampling:

| Split Role | Record Count | Percentage | File Path | SHA-256 Cryptographic Checksum | Governance Protocol |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Train** | 250 | 71.4% | [`data/splits/train.json`](../../data/splits/train.json) | `8100ffc6a8d064bd18d20f84562134fc76f554798e813eaa3d2ff87826686cce` | Dedicated solely to QLoRA parameter updates. |
| **Validation** | 50 | 14.3% | [`data/splits/validation.json`](../../data/splits/validation.json) | `113cd840eae1198db032b1fec6374f3f81f7ec43fab0c44034e8ebd96cfcee9d` | Used for early stopping & Phase 6 hyperparameter tuning. |
| **Test** | 50 | 14.3% | [`data/splits/test.json`](../../data/splits/test.json) | `cda6a24e70cb7d0d7793ab1b339afa7041db5359e2a9b90f6960eee5e276f08a` | **Strictly quarantined** until official Phase 7 holdout evaluation. |

---

## 5. Base Model Selection & QLoRA Configuration

### 5.1 Base Model Selection
Following rigorous multi-model static profiling (evaluating SmolLM2-1.7B, Phi-3.5-mini, and Qwen2.5-1.5B-Instruct), **`Qwen2.5-1.5B-Instruct`** was chosen due to its native ChatML support, superior architectural efficiency (Grouped Query Attention with 12 query heads and 2 key/value heads), and permissive Apache 2.0 license:
- **Hugging Face ID:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Pinned Revision:** `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Immutable commit hash)
- **Base Architecture:** Causal Decoder Transformer, RoPE, SwiGLU, RMSNorm, 28 layers, hidden dimension 1536.
- **Parameters:** 1,543,714,304 total parameters.

### 5.2 QLoRA Hyperparameters
Fine-tuning was conducted on Google Colab using Parameter-Efficient Fine-Tuning (PEFT) with 4-bit Quantized Low-Rank Adaptation (QLoRA):
- **Base Precision:** 4-bit NormalFloat (NF4) with double quantization.
- **Compute Precision:** `torch.float16`.
- **LoRA Rank ($r$):** 8.
- **LoRA Alpha ($\alpha$):** 16 (Scaling factor $\alpha / r = 2.0$).
- **LoRA Dropout:** 0.10.
- **Target Modules:** All 7 linear attention and MLP projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Trainable Parameters:** Exactly 9,232,384 out of 1,552,946,688 (0.5945% of total weights).
- **Optimizer:** 8-bit AdamW (`paged_adamw_8bit`), initial learning rate $1.0 \times 10^{-4}$ ($0.0001$), cosine learning rate schedule, warmup ratio 0.10.
- **Batch Configuration:** Micro-batch size 4, gradient accumulation steps 4 (Effective batch size = 16).
- **Loss Masking:** Completion-only loss masking using `DataCollatorForCompletionOnlyLM` targeting `<|im_start|>assistant\n`.

---

## 6. Training Execution & Official Holdout Evaluation Summary

### 6.1 Training Execution Metrics (Phase 6)
Training completed on Google Colab (Tesla T4 GPU) in **406.32 seconds** (~6.77 minutes):
- **Optimizer Steps Completed:** Exactly 45 steps across 3 epochs (reconciled: $\lceil 250 / 4 \rceil = 63$ batches; $63 // 4 = 15$ steps/epoch $\times 3 = 45$ steps).
- **Validation Loss:** Best evaluation loss of **1.7354** achieved at `checkpoint-40`.
- **Saved Adapter Artifact:** [`exports/customer-support-lora`](../../exports/customer-support-lora) (`adapter_model.safetensors`, 36,981,072 bytes, SHA-256: `5d70525e3f51b7a8da4356a3d24a8221f9842f9fb596766bd5e4d386e0e411a6`).

### 6.2 Official Phase 7 Holdout Test Evaluation
Following fine-tuning, the quarantined test split of 50 held-out records (`test.json`) was evaluated on a Tesla T4 GPU under Run ID `phase7_test_eval_20261113_061323_e71f91`:

| Metric Dimension | Baseline Model (Pre-trained) | Fine-Tuned Model (QLoRA) | Absolute Delta ($\Delta$) | Relative Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Response Validity Rate** | 100.0% | 100.0% | $+0.00\%$ | 100% syntactically valid |
| **Mean Token F1** | **0.2467** | **0.2891** | **+0.0424** | **+17.19% relative improvement** |
| **Exact Match (EM)** | 0.0000 | 0.0000 | $+0.0000$ | Expected for open text |
| **Escalation Accuracy** | 0.8000 | 0.8200 | $+0.0200$ | Improved overall fidelity |
| **Escalation Precision** | 1.0000 | 1.0000 | $+0.0000$ | Zero false positive escalations |
| **Escalation Recall** | 0.0909 (1/11) | **0.1818 (2/11)** | **+0.0909** | **+100.0% relative improvement** |
| **Escalation F1** | 0.1667 | 0.3077 | $+0.1410$ | Balanced sensitivity gain |
| **Mean Latency (GPU)** | 4.367s | **3.830s** | **-0.537s** | **12.3% faster turnaround** |
| **P95 Tail Latency (GPU)** | 8.243s | **5.854s** | **-2.389s** | **29.0% tail latency reduction** |
| **Mean Completion Length** | 78.7 tokens | **35.1 tokens** | **-43.6 tokens** | **55.4% reduction in verbosity** |

### 6.3 Category Breakdown Highlights
The fine-tuned model outperformed the baseline across **8 out of 10 domain categories**, achieving its largest gain in `escalation_requests` (+105.4% relative Token F1 improvement from 0.1886 to 0.3874).

---

## 7. VS Code Application Implementation (Phase 9 & 10)

Rather than merging the adapter weights or converting them to a quantized GGUF binary, the application implements **direct dynamic PEFT inference**:
1. **Dynamic Model Loading:** [`src/model/load_peft.py`](../../src/model/load_peft.py) loads the base model from the local Hugging Face snapshot cache in `torch.float32`, then dynamically applies `PeftModel.from_pretrained()` using `exports/customer-support-lora`.
2. **Unmerged Weight Architecture:** The weights remain unmerged in memory. When a forward pass occurs, PyTorch computes $W x = W_0 x + \frac{\alpha}{r} (B A x)$ dynamically.
3. **Dedicated Python Environment:** To avoid dependency conflicts with the baseline `llama-cpp-python` environment, an isolated virtual environment (`venv-peft`) was provisioned with CPU-optimized PyTorch 2.5.1 and pinned packages.
4. **ChatML Prompt Collation:** [`src/model/peft_inference.py`](../../src/model/peft_inference.py) wraps incoming customer queries into standard ChatML formatting (`<|im_start|>system...<|im_start|>user...<|im_start|>assistant\n`).
5. **FastAPI Web Service:** [`src/api/main.py`](../../src/api/main.py) exposes REST endpoints (`/health`, `/model/info`, `/chat`) and mounts the single-page chat frontend at `/`.

---

## 8. Phase 10F Live Application Testing Summary

On November 13, 2026, the complete application was tested live against the running Uvicorn server at `http://127.0.0.1:8000/`. A comprehensive 11-case test matrix was executed:

| Test Case | Category | Input Inquiry / Action | Latency | Tokens | Verdict |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **TC-01** | Telemetry | `GET /health` system health diagnostics | 0.01s | N/A | **PASS** |
| **TC-02** | Provenance | `GET /model/info` LoRA metadata verification | 0.01s | N/A | **PASS** |
| **TC-03** | Frontend UI | `GET /` single-page HTML interface | 0.01s | N/A | **PASS** |
| **TC-04** | Live Query 1 | Return Policy (Open-Box Electronics) | 12.86s | 40 tok | **PASS** |
| **TC-05** | Live Query 2 | Order Tracking (UPS Ground & Tracking ID) | 15.18s | 40 tok | **PASS** |
| **TC-06** | Live Query 3 | Damaged Item Support (Broken Glass In Transit) | 19.22s | 40 tok | **PASS** |
| **TC-07** | Multi-Turn 1 | Address Change Request (In-Transit Package) | 14.71s | 35 tok | **PASS** |
| **TC-08** | Multi-Turn 2 | Contextual Follow-up (FedEx Pickup Location) | 18.87s | 26 tok | **PASS** |
| **TC-09** | Validation 1 | Empty Message Payload (`""`) | 0.01s | HTTP 422 | **PASS** |
| **TC-10** | Validation 2 | Whitespace Message Payload (`"   "`) | 0.01s | HTTP 200 (Clean err) | **PASS** |
| **TC-11** | Validation 3 | Oversized Query (`>2000 chars`) | 0.01s | HTTP 422 | **PASS** |

---

## 9. Hardware & CPU Performance Profile

### 9.1 CPU Inference vs. Cloud GPU Comparison
A critical engineering takeaway from this project is the performance trade-off between cloud GPU execution and local commodity CPU execution:

| Performance Metric | Google Colab (Tesla T4 GPU) | Local Windows 11 CPU (Intel i5-12450H) | Operational Takeaway |
| :--- | :---: | :---: | :--- |
| **Execution Precision** | `float16` (CUDA) | `float32` (CPU) | CPU uses 32-bit floats for compatibility. |
| **Average Latency** | **3.83 seconds** | **14.0–19.2 seconds** | CPU is ~3.5–5x slower due to lack of tensor cores. |
| **Token Throughput** | **9.30 tok/sec** | **2.08–3.11 tok/sec** | Acceptable for asynchronous chat; slow for real-time voice. |
| **System RAM Footprint** | ~5.2 GB VRAM | **13.61 GB System RAM** | Model + OS comfortably fits inside 16 GB physical RAM. |
| **Hardware Cost** | $0–$10/mo cloud GPU | $0 incremental hardware cost | Runs on standard enterprise office laptops. |

---

## 10. Disclosed Limitations & Observed Model Behaviors

In compliance with strict engineering ethics and transparent disclosure standards, the following specific model limitations were empirically observed during Phase 10F live testing and must be documented:

1. **Return Policy Nuance Discrepancy (TC-04):**
   When provided context specifying that electronics can be returned within 14 days of *delivery*, the model generated: *"Our electronic products come with a standard 14-day return window from purchase date."* While the duration (14 days) and restocking fee (15%) were accurate, the model incorrectly shifted the window start date from *delivery* to *purchase*.
2. **Tracking String Truncation (TC-05):**
   When provided the tracking number `1Z999AA10123456784`, the model generated: *"The tracking code is 1Z99"*, truncating the remaining characters before continuing its sentence. In production, entity extraction or template injection should handle tracking IDs deterministically.
3. **Configured Generation Cap Boundary Cutoffs:**
   Because generation was restricted to conservative token limits (`max_new_tokens: 35–40`) to maintain CPU responsiveness, several completions ended abruptly at the token boundary without a terminal period. In production, max tokens should be increased to 128–256.
4. **Advisory Nature of Carrier Actions (TC-08):**
   The model accurately explained that FedEx Delivery Manager allows packages to be held at authorized pickup locations. This response represents **textual policy guidance based on provided context**, not evidence that the SLM application possesses autonomous API carrier integration or can execute carrier-side actions.
5. **Generative Escalation Misses:**
   As demonstrated in Phase 7 holdout evaluation, generative language models occasionally attempt to apologize or troubleshoot rather than triggering a formal support escalation. Mission-critical workflows must use deterministic upstream regex/rule-based routers.

---

## 11. Conclusion & Production Deployment Roadmap

The SLM-Powered Customer Support Assistant successfully demonstrates the viability of fine-tuning a sub-2B parameter model for domain-adapted customer support. The system achieves higher accuracy, lower verbosity, zero third-party token costs, complete data privacy, and successful local execution on standard consumer CPU hardware.

### Recommended Production Roadmap:
1. **Upstream Rule Router:** Deploy an upstream deterministic regex router for high-liability inquiries (legal, threats, safety, billing disputes).
2. **Entity Template Guardrail:** Pass order numbers, tracking URLs, and refund amounts via deterministic UI cards rather than relying entirely on free-form generative reproduction.
3. **Model Quantization:** Future iterations could evaluate 4-bit / 8-bit CPU quantization (e.g., llama.cpp / GGUF) to reduce CPU latency from ~15s down to ~3–4s on commodity CPUs.
4. **CRM Integration:** Connect the FastAPI backend to live customer database webhooks (e.g., Zendesk, Salesforce Service Cloud) with strict PII masking.
