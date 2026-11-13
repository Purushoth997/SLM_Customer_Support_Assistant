# System Architecture & Data Flow Blueprint

**Project:** SLM-Powered Customer Support Assistant  
**Repository:** `C:\slm-customer-support`  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`)  
**Adapter:** `customer-support-lora` (SHA-256: `5d70525e3f51b7a8da4356a3d24a8221f9842f9fb596766bd5e4d386e0e411a6`)  
**Document:** End-to-End Architectural Specification  

---

## 1. Architectural Overview & Component Boundaries

The system is organized into five strictly decoupled architectural tiers:
1. **Frontend Client Tier:** Single-page responsive web console providing real-time telemetry and conversational interface.
2. **FastAPI Web Service Layer:** Asynchronous REST API managing lifecycle events, request validation, and routing.
3. **Prompt Engineering & Collator Layer:** ChatML formatting, system instruction synthesis, and conversation history collation.
4. **Hugging Face PEFT Dynamic Engine:** Dynamic low-rank decomposition ($W = W_0 + \frac{\alpha}{r} B A$) executing on PyTorch CPU without weight merging.
5. **Model Storage & Checkpoint Tier:** Pinned, immutable base model weights in Hugging Face cache and verified LoRA safetensors on disk.

```mermaid
flowchart TD
    subgraph ClientTier["1. Client Tier (Browser)"]
        UI["Web Interface (index.html)"]
        HealthPill["Live Health & Status Badges"]
        ChatBox["Conversational Message Feed"]
    end

    subgraph APITier["2. API Service Layer (FastAPI / Uvicorn)"]
        Router["FastAPI Application Router (src.api.main)"]
        EndpointHealth["GET /health"]
        EndpointInfo["GET /model/info"]
        EndpointChat["POST /chat"]
        PydanticValidator["Pydantic Schema Validation (schemas.py)"]
    end

    subgraph ModelEngine["3. Dynamic PEFT Inference Engine (src.model)"]
        ChatMLCollator["ChatML Prompt Collator\n(&lt;|im_start|&gt;system...&lt;|im_start|&gt;user...)"]
        Tokenizer["Qwen2.5 AutoTokenizer (vocab_size: 151,936)"]
        PEFTRuntime["Hugging Face PEFT (peft.PeftModel)\nDynamic Forward: W_eff = W_0 + (alpha/r)*B*A"]
        PyTorchCPU["PyTorch 2.5.1 CPU Execution Engine\n(Intel i5-12450H Multi-threading)"]
        Sampler["Generation Sampler (temp=0.3, top_p=0.85, top_k=40)"]
    end

    subgraph StorageTier["4. Model Asset & Checkpoint Storage"]
        BaseModelWeights["Base Model: Qwen/Qwen2.5-1.5B-Instruct\n(Revision: 989aa798..., 1.54B Params, FP32)"]
        LoRAWeights["LoRA Adapter: customer-support-lora\n(r=8, alpha=16, 9.23M Params, Safetensors)"]
    end

    UI -->|"HTTP GET /health"| EndpointHealth
    UI -->|"HTTP GET /model/info"| EndpointInfo
    UI -->|"HTTP POST /chat {message, context, history}"| EndpointChat

    EndpointHealth --> Router
    EndpointInfo --> Router
    EndpointChat --> PydanticValidator

    PydanticValidator -->|"Validated ChatRequest"| ChatMLCollator
    ChatMLCollator -->|"Formatted ChatML String"| Tokenizer
    Tokenizer -->|"Input IDs & Attention Mask"| PEFTRuntime

    BaseModelWeights -.->|"Loaded at Startup (Frozen)"| PEFTRuntime
    LoRAWeights -.->|"Dynamically Attached (Unmerged)"| PEFTRuntime

    PEFTRuntime --> PyTorchCPU
    PyTorchCPU --> Sampler
    Sampler -->|"Generated Token Stream"| Tokenizer
    Tokenizer -->|"Decoded Text Response"| EndpointChat
    EndpointChat -->|"ChatResponse JSON {response, latency, tokens}"| ChatBox
```

---

## 2. End-to-End Request Lifecycle & Sequence Flow

When a user submits a customer support inquiry, the system processes the request through the following sequence:

```mermaid
sequenceDiagram
    autonumber
    actor Customer as User / Customer
    participant UI as Browser (index.html)
    participant API as FastAPI Router (/chat)
    participant Schema as Pydantic Validator
    participant Engine as PEFT Inference Engine
    participant Tokenizer as Qwen2.5 Tokenizer
    participant PEFT as PeftModel (Base + LoRA)
    participant CPU as PyTorch 2.5.1 CPU

    Customer->>UI: Enters inquiry + optional policy context
    UI->>UI: Displays user message bubble & starts UI timer
    UI->>API: POST /chat {message, context, history, max_tokens}
    
    API->>Schema: Validate ChatRequest payload
    alt Validation Failure (e.g. empty or >2000 chars)
        Schema-->>API: ValidationError
        API-->>UI: HTTP 422 Unprocessable Entity
        UI-->>Customer: Display error banner
    else Validation Success
        Schema-->>API: Validated ChatRequest Object
    end

    API->>Engine: generate_response(question, context, history)
    Engine->>Engine: Synthesize ChatML Prompt Template
    Note over Engine: <|im_start|>system\nYou are an Apex Support Assistant...<|im_start|>user\n...
    
    Engine->>Tokenizer: encode(chatml_prompt, return_tensors='pt')
    Tokenizer-->>Engine: input_ids (tensor), attention_mask (tensor)
    
    Engine->>PEFT: model.generate(input_ids, max_new_tokens, temperature=0.3, ...)
    PEFT->>CPU: Forward pass across 28 Transformer layers
    Note over PEFT,CPU: Dynamic linear layer computation:\nY = X * W_base + (16/8) * (X * A) * B
    CPU-->>PEFT: Raw generated token ID sequence
    PEFT-->>Engine: output_token_ids
    
    Engine->>Tokenizer: decode(output_token_ids[prompt_len:], skip_special_tokens=True)
    Tokenizer-->>Engine: generated_text_string
    Engine->>Engine: Calculate execution latency (sec) and token counts
    
    Engine-->>API: ChatResponse {response, latency, prompt_tokens, completion_tokens}
    API-->>UI: HTTP 200 OK (JSON)
    UI->>UI: Render response bubble with latency & token metrics
    UI-->>Customer: Displays customer support response
```

---

## 3. Deep Component Differentiation

### 3.1 Base Model (`Qwen/Qwen2.5-1.5B-Instruct`)
- **Role:** Foundational linguistic, grammatical, and reasoning substrate.
- **Parameters:** 1,543,714,304 parameters across 28 transformer decoder blocks.
- **State in Memory:** Loaded in full 32-bit floating point precision (`torch.float32`) on CPU. All parameters have `requires_grad = False` and are completely frozen.
- **Storage:** Stored immutably inside the user's local Hugging Face snapshot cache (`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`).
- **Cryptographic Immutability:** Locked to git commit hash `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

### 3.2 LoRA Adapter (`customer-support-lora`)
- **Role:** Domain-specific parameter delta adaptation for retail customer support dialogues.
- **Parameters:** Exactly 9,232,384 parameters (0.5945% of base model size).
- **Decomposition:**
  For each targeted linear weight matrix $W_0 \in \mathbb{R}^{d \times k}$, low-rank matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ are learned, where $r=8$ and scaling factor $\frac{\alpha}{r} = \frac{16}{8} = 2.0$.
- **Target Modules:** All 7 projection layers in the self-attention and MLP blocks:
  - Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - Feed-Forward (MLP): `gate_proj`, `up_proj`, `down_proj`
- **File Provenance:** Stored in [`exports/customer-support-lora/adapter_model.safetensors`](../../exports/customer-support-lora/adapter_model.safetensors) (36,981,072 bytes, SHA-256: `5d70525e3f51b7a8da4356a3d24a8221f9842f9fb596766bd5e4d386e0e411a6`).

### 3.3 Dynamic PEFT Runtime (Unmerged Inference)
- **Mechanism:** Implemented via Hugging Face `peft.PeftModel.from_pretrained()`.
- **Dynamic Forward Computation:**
  Unlike static weight merging (which permanently alters $W_0$ into $W_{\text{merged}} = W_0 + \Delta W$), PEFT intercepts every targeted linear module call during the forward pass:
  $$h = W_0 x + \left(\frac{\alpha}{r}\right) B (A x)$$
- **Benefits:**
  1. Base model weights remain pristine and unmodified on disk and in memory.
  2. Zero quantization or conversion loss from intermediate exports.
  3. Enables multi-tenant serving where distinct domain adapters (e.g., technical support, billing, logistics) share a single base model instance.

### 3.4 FastAPI Service & Web Interface
- **FastAPI Framework:** Implements non-blocking asynchronous request dispatch with complete Pydantic contract validation.
- **REST Endpoints:**
  - `GET /health`: System health status, memory footprint, and device telemetry.
  - `GET /model/info`: Full model provenance, adapter checksum verification, and LoRA hyperparameters.
  - `POST /chat`: Primary customer support inference endpoint with token counting and latency telemetry.
  - `GET /`: Serves the single-page chat user interface.
- **Frontend Architecture:** Vanilla JavaScript/CSS single-page application requiring no external build tools, npm packages, or third-party CDNs. Features automatic polling of `/health` and live status pills.

---

## 4. Hardware & CPU Execution Profile

### 4.1 Memory Footprint on Windows 11
- **Total Physical RAM:** 15.70 GB DDR4.
- **Operating System + Background Services:** ~7.2 GB.
- **Application Process Footprint (Uvicorn + PyTorch + Qwen2.5-1.5B FP32 + PEFT):** ~6.4 GB resident memory.
- **Total In-Use Memory During Serving:** **13.61 GB** (~86.7% utilization).
- **Available System Headroom:** ~2.1 GB free RAM, guaranteeing zero swapping to disk or pagefile thrashing.

### 4.2 CPU Multi-Threading Execution
- **Processor:** Intel Core i5-12450H (8 Cores: 4 Performance Cores, 4 Efficient Cores, 12 Threads).
- **Instruction Sets Utilized:** AVX2, FMA3, and OpenMP multi-threaded GEMM kernels inside PyTorch `torch.ops.aten`.
- **Inference Latency Profile:**
  - TTFT (Time to First Token): ~1.2–2.0 seconds.
  - Inter-token Latency: ~350–480 ms per token (~2.1–2.8 tokens/second).
  - Typical Query Latency (35–40 tokens): **12.86s to 19.22s**.
