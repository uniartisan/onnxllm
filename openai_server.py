from transformers import AutoModel, AutoTokenizer
from torch.cuda import get_device_properties
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, Request, status, HTTPException
import subprocess
import time
import sysconfig
from typing import List
import sys
import re
import json
import asyncio
import os
import gc
import uuid
from contextlib import asynccontextmanager


from utils.model import prepare_model, stream_response, generate


def init_model(modelpath):
    global tokenizer, chat_model
    chat_model, tokenizer = prepare_model(modelpath)


chatmodel_onnx = {
    "enable": True,
    "model_name": "qwen1.5-7-onnx",
    "type": "chatglm",
    "tokenizer": None,
    "device": "gpu",
    "quantize": -1,
    "model": None,
    "cache_dir": "./models",
    "max_tokens": 256,
    "init": init_model,
    "im_end": "<|im_end|>",
    "im_start": "<|im_start|>",
    "endoftext": "<|endoftext|>",
}


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_model("models\qwen1.5-7-Chat-avx2-quantizer")


def ngrok_connect():
    from pyngrok import ngrok, conf

    conf.set_default(conf.PyngrokConfig(ngrok_path="./ngrok"))
    ngrok.set_auth_token(os.environ["ngrok_token"])
    http_tunnel = ngrok.connect(8005)
    print(http_tunnel.public_url)


class Message_chatgpt(BaseModel):
    role: str
    content: dict
    parent_message_id: Optional[str] = None
    id: Optional[str] = None


class Body_chatgpt(BaseModel):
    action: Optional[str] = None
    messages: List[Message_chatgpt]
    model: Optional[str] = None
    stream: Optional[bool] = True
    max_tokens: Optional[int] = 2048
    mode: Optional[str] = None
    parent_message_id: Optional[str] = None


class Body_OpenAI(BaseModel):
    action: Optional[str] = None
    messages: List
    model: Optional[str] = None
    stream: Optional[bool] = True
    max_tokens: Optional[int] = 2048
    mode: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95


@app.get("/")
def read_root():
    return {"Hello": "World!"}


@app.post("/v1/chat/completions")
async def conversation(body: Body_OpenAI, request: Request):
    async def eval_openailike(history, top_p=0.7, temperature=0.95):
        result = {}
        result["id"] = str(uuid.uuid4())
        result["model"] = "gpt-3.5-turbo"
        result["created"] = int(time.time())
        result["object"] = "chat.completion"
        origin = ""
        response_re = ""
        choices0 = {"index": 0, "message": {"content": None, "role": "assistant"}}
        if body.stream:
            a = time.time()
            for response_re in stream_response(
                model=chat_model,
                tokenizer=tokenizer,
                history=history,
                max_output_length=max(chatmodel_onnx["max_tokens"], body.max_tokens),
                end_token=tokenizer.encode("<|endoftext|>")[0],
                im_stop_token=tokenizer.encode("<|im_end|>")[0],
                top_p=top_p,
                temperature=temperature,
            ):
                response_re = (
                    response_re.replace(chatmodel_onnx["im_start"], "")
                    .replace(chatmodel_onnx["im_end"], "")
                    .replace(chatmodel_onnx["endoftext"], "")
                )
                # print(response_re)
                if await request.is_disconnected():
                    gc.collect()
                    # print("disconnect")
                    # print(time.time() - a)
                    break

                choices0["delta"] = {"content": response_re[len(origin) :]}
                origin = response_re
                result["choices"] = [choices0]
                yield json.dumps(result)

            # 回答已经完成
            print(response_re[len(origin) :])
            choices0["delta"] = {"content": response_re[len(origin) :]}
            choices0["message"]["content"] = response_re
            result["choices"][0]["finish_reason"] = "stop"
            yield json.dumps(result, sort_keys=True)
            print(time.time() - a, response_re)
            yield "[DONE]"
        else:
            response_re = generate(
                model=chat_model,
                tokenizer=tokenizer,
                history=history,
                output_length=max(chatmodel_onnx["max_tokens"], body.max_tokens),
                end_token=tokenizer.encode("<|endoftext|>")[0],
                top_p=top_p,
                temperature=temperature,
            )
            choices0["message"]["content"] = response_re.replace(
                chatmodel_onnx["im_start"], ""
            ).replace(chatmodel_onnx["im_end"], "")
            result["choices"] = [choices0]
            result["choices"][0]["finish_reason"] = "stop"
            yield json.dumps(result, sort_keys=True)
        gc.collect()

    # print(body)
    question_messages = body.messages
    # print(question_messages)
    model = body.model

    if len(question_messages) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

    Prompt_system = "你是 Karvis，一个由 Minisforum Inc 开发的内置于 Minisforum PC 中的大型语言模型。请你尽全力回答用户的问题。"
    history = [{"role": "system", "content": Prompt_system}]

    for i in question_messages:
        history.append(i)
    print(history)

    return EventSourceResponse(
        eval_openailike(
            history,
            top_p=body.top_p,
            temperature=body.temperature,
        ),
        ping=120,
    )


if __name__ == "__main__":
    uvicorn.run(
        "openai_server:app",
        host="0.0.0.0",
        port=8005,
        reload=False,
        app_dir=".",
        timeout_keep_alive=180,
    )
