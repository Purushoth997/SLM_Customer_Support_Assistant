"""
Pydantic Request and Response Schemas for SLM Customer Support API.
Defines strict data contracts for health checks, model telemetry, and chat completions.
Includes a graceful fallback layer if Pydantic is not yet installed in the environment.
"""

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    class ChatMessage(BaseModel):
        role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
        content: str = Field(..., description="Message text content")

    class ChatRequest(BaseModel):
        message: str = Field(..., min_length=1, max_length=2000, description="Customer inquiry string")
        context: Optional[str] = Field(None, max_length=4000, description="Optional business context or policy details")
        history: Optional[List[ChatMessage]] = Field(default=None, description="Optional multi-turn conversation history")
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
        max_new_tokens: Optional[int] = Field(default=None, ge=1, le=1024, description="Maximum generation token count")
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p nucleus sampling threshold")
        top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Top-k sampling threshold")
        repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0, description="Repetition penalty")

    class ChatResponse(BaseModel):
        success: bool = Field(..., description="Whether generation succeeded")
        answer: str = Field(..., description="Assistant response text")
        latency_seconds: float = Field(..., description="Inference turnaround latency in seconds")
        prompt_tokens: int = Field(default=0, description="Count of prompt tokens")
        completion_tokens: int = Field(default=0, description="Count of completion tokens")
        total_tokens: int = Field(default=0, description="Sum of prompt and completion tokens")
        tokens_per_second: float = Field(default=0.0, description="Generation throughput in tokens/second")
        model_name: str = Field(..., description="Base model name")
        adapter_name: str = Field(..., description="Attached LoRA adapter name")
        device: str = Field(..., description="Inference compute device (cpu)")
        error: Optional[str] = Field(default=None, description="Error message if success is False")

    class HealthResponse(BaseModel):
        status: str = Field(..., description="Service status: 'healthy', 'degraded', or 'weights_missing'")
        service: str = Field(..., description="Service identifier name")
        version: str = Field(..., description="Application semantic version")
        model_loaded: bool = Field(..., description="Whether base model is loaded in memory")
        adapter_loaded: bool = Field(..., description="Whether PEFT adapter is attached")
        device: str = Field(..., description="Compute device")
        cpu_usage_percent: float = Field(..., description="Host CPU utilization percent")
        ram_used_gb: float = Field(..., description="Host memory used in GB")
        ram_total_gb: float = Field(..., description="Total physical host memory in GB")

    class ModelInfoResponse(BaseModel):
        base_model: str = Field(..., description="Hugging Face base model repository ID")
        base_model_revision: str = Field(..., description="Pinned immutable commit hash")
        adapter_name: str = Field(..., description="Name of the LoRA adapter")
        adapter_dir: str = Field(..., description="Filesystem directory of the adapter")
        expected_adapter_sha256: str = Field(..., description="Target cryptographic checksum")
        adapter_verified: bool = Field(..., description="Whether local adapter matches expected checksum")
        lora_rank: int = Field(..., description="LoRA rank (r)")
        lora_alpha: int = Field(..., description="LoRA alpha scaling factor")
        lora_dropout: float = Field(..., description="LoRA dropout rate")
        target_modules: List[str] = Field(..., description="Linear projection modules adapted by LoRA")
        dynamic_peft: bool = Field(default=True, description="True: adapter loaded dynamically without merging")
        merged_weights: bool = Field(default=False, description="Always False: weights remain unmerged")
        gguf_exported: bool = Field(default=False, description="Always False: native Transformers/PEFT inference")

except ImportError:
    # Graceful dataclass fallback if pydantic is not yet installed
    from dataclasses import dataclass, asdict

    @dataclass
    class ChatMessage:
        role: str
        content: str

        def dict(self) -> Dict[str, Any]:
            return asdict(self)

        def model_dump(self) -> Dict[str, Any]:
            return asdict(self)

    @dataclass
    class ChatRequest:
        message: str
        context: Optional[str] = None
        history: Optional[List[ChatMessage]] = None
        temperature: Optional[float] = None
        max_new_tokens: Optional[int] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None
        repetition_penalty: Optional[float] = None

        def dict(self) -> Dict[str, Any]:
            return asdict(self)

        def model_dump(self) -> Dict[str, Any]:
            return asdict(self)

    @dataclass
    class ChatResponse:
        success: bool
        answer: str
        latency_seconds: float
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        tokens_per_second: float = 0.0
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
        adapter_name: str = "customer-support-lora"
        device: str = "cpu"
        error: Optional[str] = None

        def dict(self) -> Dict[str, Any]:
            return asdict(self)

        def model_dump(self) -> Dict[str, Any]:
            return asdict(self)

    @dataclass
    class HealthResponse:
        status: str
        service: str
        version: str
        model_loaded: bool
        adapter_loaded: bool
        device: str
        cpu_usage_percent: float
        ram_used_gb: float
        ram_total_gb: float

        def dict(self) -> Dict[str, Any]:
            return asdict(self)

        def model_dump(self) -> Dict[str, Any]:
            return asdict(self)

    @dataclass
    class ModelInfoResponse:
        base_model: str
        base_model_revision: str
        adapter_name: str
        adapter_dir: str
        expected_adapter_sha256: str
        adapter_verified: bool
        lora_rank: int
        lora_alpha: int
        lora_dropout: float
        target_modules: List[str]
        dynamic_peft: bool = True
        merged_weights: bool = False
        gguf_exported: bool = False

        def dict(self) -> Dict[str, Any]:
            return asdict(self)

        def model_dump(self) -> Dict[str, Any]:
            return asdict(self)
