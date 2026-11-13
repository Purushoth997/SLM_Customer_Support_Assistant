# SLM-Powered Customer Support Assistant

An engineering-grade, domain-adapted Small Language Model (SLM) customer support automation system built with **Qwen2.5-1.5B-Instruct**. Optimized for both cloud GPU parameter-efficient fine-tuning (QLoRA) and on-premise CPU execution via Hugging Face Transformers and PEFT, backed by rigorous static audits, multi-metric benchmarking, and safety guardrails.

---

## 1. Project Overview

The **SLM-Powered Customer Support Assistant** is a task-specific language model system engineered to deliver fast, accurate, and policy-grounded responses to enterprise customer inquiries. By adapting a foundational 1.5-billion-parameter open-weights model (`Qwen2.5-1.5B-Instruct`) to the customer service domain using Low-Rank Adaptation (LoRA), the system provides enterprise-grade customer assistance while maintaining full control over inference economics, data governance, and operational infrastructure.

> **CRITICAL PRODUCTION DISCLAIMER:**  
> This project is an applied machine learning engineering benchmark and reference implementation. **It is NOT certified for unmonitored production deployment.** While fine-tuning demonstrates statistically significant gains in lexical alignment (+17.2% Token F1) and latency reduction (-12.3% on GPU), Small Language Models remain susceptible to edge-case hallucinations and generative escalation misses. Mission-critical customer routing must be paired with upstream deterministic guardrails.

---

## 2. Problem Statement & Proposed Solution

### The Challenge of Frontier LLMs
Enterprise support centers frequently rely on cloud-hosted frontier LLMs (e.g., GPT-4o, Claude 3.5 Sonnet) for customer support automation. While capable, this approach introduces significant operational drawbacks:
- **Recurring Token Costs:** High per-token pricing scales aggressively with customer support ticket volume.
- **Latency & Availability Variances:** External network round-trips and provider rate limits create unpredictable response times.
- **Verbose & Preamble-Heavy Outputs:** General-purpose models frequently generate conversational filler rather than concise, actionable instructions.
- **Data Privacy & Compliance Exposure:** Transmitting proprietary customer identifiers, order records, and company policies across third-party APIs introduces regulatory risks (GDPR, CCPA, SOC 2).

### The Small Language Model (SLM) Solution
By fine-tuning a sub-2-billion parameter model on domain-specific support dialogues, this project demonstrates that an SLM can achieve:
- **Zero Third-Party Token Fees:** Predictable, fixed-cost infrastructure without usage billing.
- **100% Data Privacy:** Complete on-premise execution; customer data never leaves local/enterprise boundaries.
- **Disciplined Tone & Reduced Verbosity:** A 55.4% reduction in wordy filler compared to the base model.
- **Consumer Hardware Feasibility:** Fully runnable on standard CPU workstations (16 GB RAM) without requiring expensive dedicated inference GPUs.

---

## 3. Key Features

- **Dynamic PEFT LoRA Loading:** Dynamically loads unmerged LoRA adapter weights ($r=8, \alpha=16$) onto the frozen base model at runtime, preserving base model immutability and enabling multi-adapter flexibility.
- **ChatML Prompt Collation:** Native formatting using ChatML tokens (`<|im_start|>system...<|im_start|>user...<|im_start|>assistant\n`).
- **Production-Grade FastAPI Service:** Asynchronous REST API providing health diagnostics (`GET /health`), model provenance (`GET /model/info`), and inference (`POST /chat`) with Pydantic request validation.
- **Single-Page Chat Console:** Lightweight, self-contained HTML5/JavaScript user interface featuring real-time health polling and telemetry badges.
- **Quarantined Evaluation Methodology:** Rigorous 70/15/15 dataset partitioning with cryptographic checksums and complete isolation of the held-out test split.
- **Exhaustive Technical Documentation:** Full suite of mentor submission documents, architecture diagrams, and demonstration scripts.

---

## 4. System Architecture & Data Flow

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
|  FastAPI / Uvicorn / Pydantic Contracts (src/api/main.py, schemas.py)            |
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
                       PyTorch 2.5.1 CPU Inference Engine
                       Dynamic Forward: W_eff * x = W_0 * x + (alpha/r) * B * A * x
```

For complete architecture details and Mermaid diagrams, see [`reports/phase10g_final_submission/FINAL_ARCHITECTURE.md`](reports/phase10g_final_submission/FINAL_ARCHITECTURE.md).

### Repository Layout
```text
slm-customer-support/
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Baseline CPU inference dependencies
├── requirements-cloud.txt             # Cloud GPU QLoRA training dependencies
├── requirements-peft.txt              # Isolated VS Code CPU PEFT application dependencies
├── .gitignore                         # Exclusions for weights, datasets, venvs, and cache
│
├── configs/                           # Application and training configurations
│   ├── model_config.yaml              # Base model metadata & CPU inference settings
│   ├── training_config.yaml           # QLoRA fine-tuning hyperparameters
│   ├── evaluation_config.yaml         # Benchmark settings & metric definitions
│   └── dataset_config.yaml            # Schema validation & category definitions
│
├── notebooks/                         # Google Colab workflows (clean templates)
│   ├── phase6_colab_preflight.ipynb   # GPU environment & dependency preflight
│   ├── phase6_qlora_training.ipynb    # Reference QLoRA training notebook
│   ├── phase6_model_comparison_evaluation.ipynb # Baseline vs Fine-tuned validation comparison
│   └── phase7_final_holdout_evaluation.ipynb    # Final holdout test evaluation
│
├── reports/                           # Technical audit reports and submissions
│   └── phase10g_final_submission/     # Complete mentor submission package
│       ├── FINAL_PROJECT_REPORT.md    # Comprehensive 11-section project report
│       ├── PROJECT_DEMONSTRATION_SCRIPT.md # 2.5–3 min mentor walkthrough guide
│       ├── FINAL_ARCHITECTURE.md      # Architectural specification with Mermaid diagrams
│       ├── FINAL_SUBMISSION_CHECKLIST.md   # Audit checklist and verification log
│       └── PHASE10G_FINAL_AUDIT.md    # Itemized technical audit matrix
│
├── scripts/                           # Operational scripts
│   ├── download_base_model.py         # Pinned base model snapshot downloader
│   ├── verify_peft_assets.py          # Cryptographic asset verification script
│   └── run_smoke_tests.py             # Smoke testing utility
│
├── src/                               # Modular application source code
│   ├── config.py                      # Centralized configuration & environment loader
│   ├── api/                           # FastAPI service layer & schemas
│   │   ├── main.py                    # API entry point & Uvicorn runner
│   │   ├── schemas.py                 # Pydantic request/response schemas
│   │   └── static/index.html          # Single-page chat interface
│   ├── data/                          # Dataset preprocessing & collation
│   ├── evaluation/                    # Metrics calculation & benchmark harnesses
│   ├── model/                         # Dynamic PEFT model loader & inference engine
│   │   ├── load_peft.py               # Dynamic Hugging Face PEFT loader
│   │   └── peft_inference.py          # ChatML generation & telemetry service
│   └── optimization/                  # Inference profiling utilities
│
└── tests/                             # Application unit tests
    └── test_peft_application.py       # Integration & schema validation tests
```

---

## 5. Technology Stack

- **Core ML Frameworks:** PyTorch 2.5.1 (CPU build), Hugging Face Transformers 4.48.3, PEFT 0.14.0, Accelerate 1.3.0, Safetensors 0.5.2.
- **API & Web Service:** FastAPI 0.115.8, Uvicorn 0.34.0, Starlette 0.45.3, Pydantic 2.10.6.
- **Frontend:** Vanilla HTML5, modern CSS, asynchronous JavaScript (Fetch API). Zero external npm dependencies or CDN links.
- **Cloud Training Stack:** Google Colab, NVIDIA Tesla T4 GPU (16 GB VRAM), CUDA 12.4, BitsAndBytes 0.45.3 (4-bit NF4 quantization).

---

## 6. Dataset Methodology & Governance

To guarantee absolute privacy and prevent data contamination, zero real customer PII was used. A high-quality synthetic customer support dataset comprising **350 records** was engineered for the fictional enterprise *"Apex Retail Solutions"*.

> **DATASET EXCLUSION NOTICE:**  
> In accordance with privacy and quarantine governance, raw dataset files and split records are **excluded from this repository** via `.gitignore`. The dataset structure, schema, and cryptographic hashes are documented below for complete auditability.

### A. Domain Categories
The dataset is balanced across 10 core customer support domains (exactly 35 records per category):
1. `order_status`: Tracking, transit updates, address modifications, warehouse exceptions.
2. `shipping_delivery`: Delivery speeds, courier delays, international customs, damaged boxes.
3. `payment_issues`: Split tender, expired gift cards, double charges, currency policies.
4. `refund_requests`: Refund timelines, non-refundable digital keys, price match adjustments.
5. `return_policies`: 30-day windows, return condition requirements, international return duties.
6. `product_information`: Hardware compatibility, dimensions, technical specifications.
7. `account_support`: Password resets, 2FA recovery, profile updates, account deletions.
8. `customer_complaints`: Damaged items, poor service conduct, delivery exceptions.
9. `escalation_requests`: Explicit supervisor transfers, legal claims, vendor disputes.
10. `general_support`: Store locations, operating hours, phone assistance.

### B. Split Isolation & Cryptographic Checksums
The 350 cleaned records were partitioned into a leakage-free 70/15/15 split using stratified category sampling:

| Split Role | Record Count | Allocation | SHA-256 Cryptographic Checksum | Governance Role |
| :--- | :---: | :---: | :--- | :--- |
| **Train** | 250 | 71.4% | `8100ffc6a8d064bd18d20f84562134fc76f554798e813eaa3d2ff87826686cce` | Used solely for QLoRA parameter updates. |
| **Validation** | 50 | 14.3% | `113cd840eae1198db032b1fec6374f3f81f7ec43fab0c44034e8ebd96cfcee9d` | Used for early stopping & development tuning. |
| **Test** | 50 | 14.3% | `cda6a24e70cb7d0d7793ab1b339afa7041db5359e2a9b90f6960eee5e276f08a` | **Strictly quarantined** until Phase 7 final holdout evaluation. |

---

## 7. Base Model & QLoRA Fine-Tuning Configuration

### Base Model
- **Identifier:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Pinned Commit Revision:** `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Immutable)
- **License:** Apache 2.0 (Permissive open-source commercial & research reuse)
- **Architecture:** Causal Decoder Transformer, RoPE, SwiGLU, RMSNorm, Grouped Query Attention (12 query heads, 2 KV heads), 28 layers, hidden dimension 1536.
- **Parameters:** 1.54 Billion total (1.31B non-embedding).

### QLoRA Hyperparameters
- **Base Precision:** 4-bit NormalFloat (NF4) with double quantization.
- **Compute Precision:** `torch.float16`.
- **LoRA Rank ($r$):** 8.
- **LoRA Alpha ($\alpha$):** 16 (Scaling factor $\alpha / r = 2.0$).
- **LoRA Dropout:** 0.10.
- **Target Modules:** All 7 linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Trainable Parameters:** 9,232,384 out of 1,552,946,688 (0.5945% of total weights).
- **Optimization:** AdamW 8-bit (`paged_adamw_8bit`), learning rate $1.0 \times 10^{-4}$, cosine schedule, warmup ratio 0.10.
- **Batching & Accumulation:** Micro-batch size 4, gradient accumulation steps 4 (Effective batch size = 16).
- **Training Duration:** 406.32 seconds (~6.77 minutes) across 45 optimizer steps on a single Tesla T4 GPU.
- **Best Checkpoint:** `checkpoint-40` (`best_eval_loss: 1.7354`).
- **Adapter Artifact:** `customer-support-lora` (SHA-256: `5d70525e3f51b7a8da4356a3d24a8221f9842f9fb596766bd5e4d386e0e411a6`).

---

## 8. Official Phase 7 Holdout Test Evaluation

The definitive benchmark was conducted on the **50 quarantined held-out test records** (`test.json`) on Google Colab (Tesla T4 GPU, FP16) under Run ID `phase7_test_eval_20261113_061323_e71f91`:

| Metric Dimension | Baseline Model | Fine-Tuned Model | Delta ($\Delta$) | Relative Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Response Validity Rate** | 100.0% | 100.0% | $+0.00\%$ | 100% syntactically valid |
| **Mean Token F1** | **0.2467** | **0.2891** | **+0.0424** | **+17.19% relative improvement** |
| **Exact Match (EM)** | 0.0000 | 0.0000 | $+0.0000$ | Expected for open-ended text |
| **Escalation Accuracy** | 0.8000 | 0.8200 | $+0.0200$ | Improved overall fidelity |
| **Escalation Precision** | 1.0000 | 1.0000 | $+0.0000$ | **Zero false alarms** |
| **Escalation Recall** | 0.0909 (1/11) | **0.1818 (2/11)** | **+0.0909** | **+100.0% relative improvement** |
| **Escalation F1** | 0.1667 | 0.3077 | $+0.1410$ | Balanced sensitivity gain |
| **Mean Latency (GPU)** | 4.367s | **3.830s** | **-0.537s** | **12.3% faster turnaround** |
| **P95 Tail Latency (GPU)** | 8.243s | **5.854s** | **-2.389s** | **29.0% tail latency reduction** |
| **Mean Completion Length** | 78.7 tok | **35.1 tok** | **-43.6 tok** | **55.4% reduction in verbosity** |

### Category-Level Performance (Holdout Test Split)
The fine-tuned model improved performance in **8 out of 10 domain categories**, led by `escalation_requests` (+105.4% relative gain from 0.1886 to 0.3874).

---

## 9. Local Model & Adapter Placement Requirements

> **MODEL ASSET ACQUISITION NOTICE:**  
> To comply with GitHub repository size limits and prevent committing multi-gigabyte binary files, **neither the base model weights (~3.1 GB) nor the LoRA adapter weights (37 MB) are stored in this Git repository.**  
> Users must acquire both assets and place them locally as instructed below before launching the application.

### Required Asset Locations
1. **Base Model Snapshot:**
   The application loads `Qwen/Qwen2.5-1.5B-Instruct` at pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` from the standard Hugging Face hub cache:
   ```powershell
   python scripts/download_base_model.py --confirm-download
   ```
2. **LoRA Adapter Checkpoint:**
   Place the trained adapter directory at:
   `exports/customer-support-lora/`
   
   Ensure the following 5 files are present:
   - `adapter_config.json`
   - `adapter_model.safetensors` (Expected SHA-256: `5d70525e3f51b7a8da4356a3d24a8221f9842f9fb596766bd5e4d386e0e411a6`)
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `special_tokens_map.json`
3. **Verify Asset Readiness:**
   ```powershell
   python scripts/verify_peft_assets.py
   ```

---

## 10. VS Code Setup & Installation Guide

### Prerequisites
- Windows 10/11, macOS, or Linux.
- Python 3.11.9 installed.
- 16 GB system RAM (recommended for CPU inference).

### Step 1: Create an Isolated Virtual Environment
```powershell
# Create dedicated virtual environment
python -m venv venv-peft

# Activate virtual environment (Windows PowerShell)
.\venv-peft\Scripts\activate
```

### Step 2: Install CPU-Optimized PyTorch & Dependencies
```powershell
# Install PyTorch CPU-only build (avoids downloading 2+ GB of unused CUDA binaries)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining application dependencies
pip install -r requirements-peft.txt
```

### Step 3: Run Verification Suite
```powershell
# Verify model assets
python scripts/verify_peft_assets.py

# Run application unit tests
python -m unittest tests.test_peft_application
```

---

## 11. Running the Application & API Endpoints

### Starting the Server
From the project root with `venv-peft` activated:
```powershell
# Start Uvicorn development server
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### Interactive Web UI
Open your browser and navigate to:
**`http://127.0.0.1:8000/`** (or `http://127.0.0.1:8000/ui`)

### REST API Endpoints
- **`GET /health`**: Host telemetry, CPU load, memory utilization, and model readiness.
  ```json
  {
    "status": "healthy",
    "service": "Apex Retail Solutions — SLM Customer Support",
    "version": "1.0.0",
    "model_loaded": true,
    "adapter_loaded": true,
    "device": "cpu",
    "ram_used_gb": 13.61,
    "ram_total_gb": 15.7
  }
  ```
- **`GET /model/info`**: Model identity, pinned revision, adapter verification, and LoRA hyperparameters.
  ```json
  {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "base_model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "adapter_name": "customer-support-lora",
    "dynamic_peft": true,
    "merged_weights": false,
    "lora_rank": 8,
    "lora_alpha": 16
  }
  ```
- **`POST /chat`**: Submit customer inquiries with optional context and conversation history.
  ```json
  // Request
  {
    "message": "What is your return policy for open-box electronics?",
    "context": "Electronics can be returned within 14 days of delivery. 15% restocking fee applies.",
    "max_tokens": 128
  }
  
  // Response
  {
    "success": true,
    "response": "Our electronics come with a 14-day return window...",
    "latency_seconds": 12.86,
    "prompt_tokens": 356,
    "completion_tokens": 40,
    "tokens_per_second": 3.11
  }
  ```

---

## 12. Phase 10F Live Application Testing Results

On November 13, 2026, the complete application was tested live against the running Uvicorn server on `http://127.0.0.1:8000/`:

| Test ID | Category | Inquiry / Action | Latency | Tokens | Status |
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

## 13. CPU Hardware Requirements & Performance Profile

| Performance Dimension | Google Colab (Tesla T4 GPU) | Local Windows 11 CPU (Intel i5-12450H) | Operational Takeaway |
| :--- | :---: | :---: | :--- |
| **Compute Precision** | `float16` (CUDA) | `float32` (CPU) | CPU uses 32-bit floats for compatibility. |
| **Average Latency** | **3.83 seconds** | **12.86–19.22 seconds** | CPU is ~3.5–5x slower due to lack of tensor cores. |
| **Throughput** | **9.30 tok/sec** | **2.08–3.11 tok/sec** | Adequate for asynchronous web chat; slow for voice. |
| **System RAM Footprint**| ~5.2 GB VRAM | **13.61 GB System RAM** | Model + OS comfortably fits inside 16 GB physical RAM. |
| **Infrastructure Cost** | $0–$10/mo cloud GPU | $0 incremental hardware cost | Runs on standard enterprise office laptops. |

---

## 14. Disclosed Limitations & Observed Nuances

In accordance with rigorous engineering ethics, the following empirically observed limitations from Phase 10F live testing are disclosed:

1. **Return Policy Window Anchor (TC-04):** When context stated that electronics can be returned within 14 days of *delivery*, the model generated: *"Our electronic products come with a standard 14-day return window from purchase date."* While the duration (14 days) and restocking fee (15%) were accurate, the model incorrectly shifted the window start date from *delivery* to *purchase*.
2. **Tracking Number Truncation (TC-05):** When provided the tracking number `1Z999AA10123456784`, the model generated: *"The tracking code is 1Z99"*, truncating the remaining characters. In production, deterministic entity injection should format tracking identifiers.
3. **Generation Token Cap Boundary:** Due to conservative generation limits (`max_new_tokens: 35–40`) used to maintain CPU responsiveness, several completions ended at the token cap before complete sentence punctuation.
4. **Advisory Nature of Carrier Guidance (TC-08):** The model accurately advised the customer on using FedEx Delivery Manager to hold packages for pickup. This response represents **textual policy guidance based on provided context**, not evidence that the SLM can autonomously execute carrier-side API actions.
5. **Generative Escalation Imperfections:** In holdout evaluation, generative regex detection achieved 0.1818 recall (2/11 detected). High-liability inquiries (safety hazards, legal threats, fraud) must be intercepted by an upstream deterministic rule-based router.

---

## 15. Safety Restrictions & Execution Locks

All six project execution safety locks remain strictly enforced in the repository:

```python
RUN_MANUAL_TRAINING = False
RUN_MODEL_MERGING = False
RUN_GGUF_EXPORT = False
RUN_HOLDOUT_TEST_EVALUATION = False
ALLOW_OVERWRITE_OUTPUTS = False
ALLOW_OVERWRITE_MODELS = False
```

---

## 16. License & Citation

- **Base Model License:** Apache 2.0 ([Qwen2.5 License](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)).
- **Project License:** MIT License for application code, architecture specifications, and documentation.
