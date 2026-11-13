"""
FastAPI Application Entry Point for SLM Customer Support Assistant.
Exposes REST endpoints for health checks, model telemetry, and chat completions
powered by Hugging Face Transformers and the customer-support-lora PEFT adapter.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    APP_TITLE,
    APP_VERSION,
    SERVER_HOST,
    SERVER_PORT,
    BASE_MODEL_NAME_OR_PATH,
    BASE_MODEL_REVISION,
    ADAPTER_DIR,
    EXPECTED_ADAPTER_SHA256,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    DEVICE,
)
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.model.load_peft import (
    check_model_assets_readiness,
    is_model_loaded,
    verify_adapter_integrity,
)
from src.model.peft_inference import generate_peft_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# FastAPI Application Factory
try:
    from fastapi import FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description="Local SLM Customer Support Service running Qwen2.5-1.5B-Instruct with fine-tuned LoRA PEFT adapter.",
    )

    # CORS configuration for local frontend communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    STATIC_DIR = Path(__file__).resolve().parent / "static"

    @app.on_event("startup")
    async def startup_event():
        """Preload the PEFT LoRA model into memory during server startup."""
        logger.info("Initializing and preloading PEFT LoRA model onto CPU...")
        try:
            from src.model.load_peft import get_peft_model
            get_peft_model()
            logger.info("PEFT LoRA model preloaded and ready for inference.")
        except Exception as e:
            logger.warning("Model preloading deferred or encountered an error: %s", e)

    @app.get("/", response_class=FileResponse, tags=["Frontend"])
    async def serve_ui():
        """Serve the local single-page customer chat interface."""
        index_file = STATIC_DIR / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="Frontend index.html not found.")
        return FileResponse(str(index_file))

    @app.get("/ui", response_class=FileResponse, tags=["Frontend"])
    async def serve_ui_alias():
        """Alias route for the chat interface."""
        return await serve_ui()

    @app.get("/health", response_model=HealthResponse, tags=["Telemetry"])
    async def health_check():
        """
        System telemetry and health diagnostics endpoint.
        Returns host memory utilization, device status, and model asset readiness.
        """
        import psutil

        readiness = check_model_assets_readiness()
        loaded = is_model_loaded()

        # Determine operational status
        if loaded:
            service_status = "healthy"
        elif readiness["adapter_verified"] and readiness["base_model_available_locally"]:
            service_status = "ready_to_load"
        elif len(readiness["missing_dependencies"]) > 0:
            service_status = "dependencies_missing"
        else:
            service_status = "weights_missing"

        vm = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(interval=None)

        return HealthResponse(
            status=service_status,
            service=APP_TITLE,
            version=APP_VERSION,
            model_loaded=loaded,
            adapter_loaded=loaded,
            device=DEVICE,
            cpu_usage_percent=float(cpu_usage),
            ram_used_gb=round(vm.used / (1024 ** 3), 2),
            ram_total_gb=round(vm.total / (1024 ** 3), 2),
        )

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["Telemetry"])
    async def model_info():
        """
        Detailed model provenance and LoRA architectural parameters.
        Confirms weights are unmerged and running native PEFT.
        """
        adapter_valid, _, details = verify_adapter_integrity()

        return ModelInfoResponse(
            base_model=BASE_MODEL_NAME_OR_PATH,
            base_model_revision=BASE_MODEL_REVISION,
            adapter_name="customer-support-lora",
            adapter_dir=str(Path(ADAPTER_DIR).resolve()),
            expected_adapter_sha256=EXPECTED_ADAPTER_SHA256,
            adapter_verified=adapter_valid,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            dynamic_peft=True,
            merged_weights=False,
            gguf_exported=False,
        )

    @app.post("/chat", response_model=ChatResponse, tags=["Inference"])
    async def chat_completion(request: ChatRequest):
        """
        Generate a customer support assistant response for a customer inquiry.
        Accepts message, optional business context, and conversation history.
        """
        history_list = None
        if request.history:
            history_list = [
                {"role": turn.role, "content": turn.content} for turn in request.history
            ]

        res = generate_peft_response(
            question=request.message,
            context=request.context,
            history=history_list,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )

        return ChatResponse(
            success=res["success"],
            answer=res["answer"],
            latency_seconds=res["latency_seconds"],
            prompt_tokens=res["prompt_tokens"],
            completion_tokens=res["completion_tokens"],
            total_tokens=res["total_tokens"],
            tokens_per_second=res["tokens_per_second"],
            model_name=res["model_name"],
            adapter_name=res["adapter_name"],
            device=res["device"],
            error=res.get("error"),
        )

except ImportError:
    # Minimal fallback representation if FastAPI is not yet installed
    app = None
    logger.warning("FastAPI framework is not installed in the active environment. Install requirements-peft.txt to run server.")


def start_server():
    """Start the Uvicorn ASGI server."""
    try:
        import uvicorn
        logger.info("Starting SLM Customer Support API on %s:%d", SERVER_HOST, SERVER_PORT)
        uvicorn.run("src.api.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
    except ImportError:
        logger.error("uvicorn is not installed. Please run: pip install -r requirements-peft.txt")
        sys.exit(1)


if __name__ == "__main__":
    start_server()
