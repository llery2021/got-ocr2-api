import ssl
import time
import os
import ctypes
import argparse
import uvicorn
import uuid
import torch
import llama_cpp
import llama_cpp.llava_cpp as llava_cpp
import llama_cpp.llava_cpp
from llama_cpp import Llama, StoppingCriteriaList
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from http import HTTPStatus
from vary_b import build_vary_vit_b as build_GOT_vit_b
from prompt_utils import get_stop_token_id, get_got_prompts, merge_embeddings

LLAMA_CPP_VERSION = "0.3.5"
TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
global engine

model_path = None
tokenizer = None
device = "cuda"
dtype = torch.bfloat16

token_embedding = None
vision_tower_high = None
mm_projector_vary = None


class ErrorResponse:
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


def create_error_response(status: HTTPStatus, message: str, error_type='invalid_request_error'):
    return JSONResponse(ErrorResponse(message=message, type=error_type, code=status.value).model_dump(), status_code=status.value)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/version")
async def show_version():
    ver = {"version": LLAMA_CPP_VERSION}
    return JSONResponse(content=ver)


@app.post("/v1/ocr")
async def ocr_v1(raw_request: Request) -> Response:
    request_id = f"llamacpp-{uuid.uuid4().hex}"

    request_dict = await raw_request.json()
    got_params_image_file = request_dict.pop("image_file")
    got_params_type = request_dict.pop("type", "ocr")
    got_params_box = request_dict.pop("box", None)
    got_params_max_tokens = request_dict.pop("max_tokens", 2048)
    got_params_top_p = request_dict.pop("top_p", 0)
    got_params_temperature = request_dict.pop("temperature", 0)

    prompt_inputs, image_tensors = get_got_prompts(
        got_params_image_file, got_params_type, got_params_box)
    created_time = int(time.time())
    choices = []
    prompt_tokens = 0
    completion_tokens = 0
    token_ids = {int(get_stop_token_id()), }

    def stop_on_token_ids(tokens, *args, **kwargs):
        return tokens[-1] in token_ids

    for i, messages in enumerate(prompt_inputs):
        input_ids = engine.tokenizer_.tokenize(prompt_inputs[i].encode())
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        images = image_tensors[i].unsqueeze(0).to(dtype=dtype, device=device)
        image_sizes = torch.tensor(
            [images.shape[0]], dtype=torch.int32, device=device)
        positions = []
        for index in range(len(input_ids)):
            positions.append(index)
        positions = torch.tensor(positions, dtype=torch.int32, device=device)
        embeddings = merge_embeddings(token_embedding, vision_tower_high,
                                      mm_projector_vary, positions, input_ids, images, image_sizes)
        embeddings = embeddings.reshape(-1).tolist()

        n_past = ctypes.c_int(engine.n_tokens)
        n_past_p = ctypes.pointer(n_past)
        FloatArray = ctypes.c_float * len(embeddings)
        embed = llava_cpp.llava_image_embed(
            embed=FloatArray(*embeddings), n_image_pos=286)
        with llama_cpp.suppress_stdout_stderr(disable=True):
            llava_cpp.llava_eval_image_embed(
                engine.ctx, embed, engine.n_batch, n_past_p,)
        engine.n_tokens = n_past.value
        prompt = engine.input_ids[: engine.n_tokens].tolist()
        stopping_criteria = StoppingCriteriaList([stop_on_token_ids])
        t1 = time.time()
        outputs = engine(prompt=prompt, top_p=got_params_top_p, max_tokens=got_params_max_tokens,
                         temperature=got_params_temperature, stop=[], stopping_criteria=stopping_criteria)
        t2 = time.time()
        print(f"推理耗时={t2-t1}s")
        choice = outputs['choices'][0]
        text = choice['text'].strip()
        ret = {"index": i,
               "text": text,
               "logprobs": choice["logprobs"],
               "finish_reason": choice["finish_reason"],
               "matched_stop": None,
               "prompt_tokens_details": None,
               }
        choices.append(ret)
        prompt_tokens += len(positions)
        completion_tokens += len(text)
        engine.reset()
        llama_cpp.llama_kv_cache_clear(engine.ctx)

    result = {"id": request_id,
              "object": "text_completion",
              "created": created_time,
              "model": model_path,
              "choices": choices,
              "usage": {
                  "prompt_tokens": prompt_tokens,
                  "total_tokens": prompt_tokens + completion_tokens,
                  "completion_tokens": completion_tokens
              }}

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    args = parser.parse_args()

    model_path = args.model_path.replace("\\", "/")

    engine = Llama(
        model_path=model_path,
        flash_attn=True,
        n_gpu_layers=-1,
        # vocab_only=True,
        seed=0,
        n_ctx=2048,
        n_batch=512
    )
    engine.set_cache(None)

    token_embedding = torch.nn.Embedding(
        151860, 1024, 151643).to(device=device, dtype=dtype)
    vision_tower_high = build_GOT_vit_b().to(device=device, dtype=dtype)
    mm_projector_vary = torch.nn.Linear(
        1024, 1024).to(device=device, dtype=dtype)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    token_embedding.weight.data = torch.load(
        f"{current_directory}/weights/embedding_weights.pt")
    vision_tower_high.load_state_dict(torch.load(
        f"{current_directory}/weights/vision_tower_high.params"))
    mm_projector_vary.load_state_dict(torch.load(
        f"{current_directory}/weights/mm_projector_vary.params"))

    app.root_path = args.root_path
    for route in app.routes:
        if not hasattr(route, 'methods'):
            continue
        methods = ', '.join(route.methods)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
