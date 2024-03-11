import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, Request, status, HTTPException
import time
from typing import List
import threading
import json
import os
import gc
import uuid
from contextlib import asynccontextmanager


from utils.model import prepare_model, stream_response, generate

chat_model = None
tokenizer = None
last_used_time = None
unload_timer = 3000  # Unload time in seconds (5 minutes)
stop_threads = False
import os
current_dir = os.getcwd()

def init_model(modelpath):
    global tokenizer, chat_model, last_used_time
    if chat_model is None:
        chat_model, tokenizer = prepare_model(modelpath)
        print("Model loaded")
    last_used_time = time.time()
    


def unload_model_background():
    global chat_model, tokenizer
    global stop_threads
    while True:
        if stop_threads: 
            break
        time.sleep(3)
        if chat_model is not None and time.time() - last_used_time > unload_timer:
            del chat_model, tokenizer
            chat_model = None
            tokenizer = None
            print("Model unloaded")
            gc.collect()  # Consider using del chat_model if necessary
         
        




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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global stop_threads
    init_model(os.path.join(r"models\\qwen"))
    # Start the background task in a separate asyncio task
    unload_thread = threading.Thread(target=unload_model_background)
    unload_thread.start()
    yield    

    # Ensure the thread is no longer running at program exit
    stop_threads = True
    gc.collect()
    


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




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
    if chat_model is None:
        init_model(os.path.join(r"models\\qwen"))

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
                    response_re
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
            choices0["message"]["content"] = response_re
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

    Prompt_system = f"""I'm Karvis, a large language model integrated into Minisforum(铭凡) PC by Minisforum(铭凡) Inc.
    Minisforum is known for its high-performance mini computers.
    The Minisforum V3 is a 14-inch Windows 3-in-1 tablet powered by the AMD Ryzen 7 8840U processor with 8 cores and 16 threads,
      a maximum boost frequency of 5.1GHz, and integrated Radeon 780M graphics. The AI computing power(NPU) is up to 16 trillion TOPS.
    The screen has a resolution of 14 inch 2560×1600, 100% DCI-P3 wide color gamut, 165Hz refresh rate, and 500 nits high brightness. 
    It supports the MPP2.6 protocol and a professional inspiration stylus with 4096 levels of pressure sensitivity.
    The body is made of magnesium alloy and weighs about 946g. It is about 9.8mm thick and is equipped with a dual fan and four heat pipe cooling system.
    It supports face recognition and fingerprint recognition, 4 speakers, and dual USB4, USB-C DP-in, SD card slot, and 3.5mm headphone jack. 
    As embedded in the V3 tablet, I'm here to provide comprehensive answers to users' inquiries."""
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
