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
    "max_tokens": 1024,
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

init_model("qwen1.5-1.8-onnx")


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
    async def eval_openailike(input, history, top_p=0.7, temperature=0.95):
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
                input_text=input,
                previous_output_text=history,
                max_output_length=max(chatmodel_onnx["max_tokens"], body.max_tokens),
                end_token=tokenizer.encode("<|endoftext|>")[0],
                top_p=top_p,
                temperature=temperature,
            ):
                response_re = response_re.replace(chatmodel_onnx['im_start'], "").replace(chatmodel_onnx['im_end'], "")
                # print(response_re)
                if await request.is_disconnected():
                    gc.collect()
                    # print("disconnect")
                    # print(time.time() - a)
                    return

                choices0["delta"] = {"content": response_re[len(origin) :]}
                origin = response_re
                result["choices"] = [choices0]
                yield json.dumps(result)
            
            # 回答已经完成
            print(response_re[len(origin):])
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
                input_text=input,
                previous_output_text=history,
                output_length=max(chatmodel_onnx["max_tokens"], body.max_tokens),
                end_token=tokenizer.encode("<|endoftext|>")[0],
                top_p=top_p,
                temperature=temperature,
            )
            choices0["message"]["content"] = response_re.replace(chatmodel_onnx['im_start'], "").replace(chatmodel_onnx['im_end'], "")
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

    user_question = ""

    history = ""
    Prompt_system = ">> System: 你是 Karvis，一个内置于 Minisforum PC 中的大型语言模型。你被设计为陪伴用户，解决用户的问题。以下是用户和你的对话：\n" 
    history += Prompt_system

    if question_messages[-1]["role"] == "user":
        user_question = question_messages[-1]["content"]

    lenth = len(question_messages)
    if lenth == 1:
        pass
    elif lenth == 2:
        # history = _history
        history += ">> User: " + question_messages[0]["content"]
        # user_question = ">> User: " + question_messages[-1]["content"]
    else:
        # assert (lenth % 2 == 0)
        # history = _history
        history += ">> User: " + question_messages[0]["content"]
        for i in range(1, lenth - 1):
            message = question_messages[i]
            # print(message)
            if message["role"] == "user":
                history_input = message["content"]
            elif message["role"] == "system" or message["role"] == "assistant":
                history_answer = message["content"]

            if i % 2 == 0:
                history += ">> User: " + history_input
                history += ">> Assistant: " + history_answer

    return EventSourceResponse(
        eval_openailike(
            user_question + ">> Assistant: ", history, top_p=body.top_p, temperature=body.temperature
        ),
        ping=360,
    )


# @app.post("/api/conversation")
# async def conversation(body: Body_chatgpt, request: Request):
#     async def eval_chatgptlike_chatglm(input, history, top_p=0.7, temperature=0.95):
#         result = {}
#         result["id"] = str(uuid.uuid4())
#         # result["parent_message_id"] = id
#         # "parent_message_id": id
#         response = {"content": None,
#                     "id": result["id"], "create_time": None, "update_time": None, "author": {
#                         "role": "assistant",
#                         "name": None,
#                         "metadata": {}
#                     },  "content": {
#                         "content_type": "text",
#                         "parts": None
#                     }, }

#         if body.stream:
#             for response_re, _ in chat_model.stream_chat(tokenizer, input, history,
#                                                             max_length=max(
#                                                                 chatmodel_onnx["max_tokens"], body.max_tokens),
#                                                             top_p=top_p,
#                                                             temperature=temperature):
#                 if await request.is_disconnected():
#                     gc.collect()
#                     return

#                 response["content"]["parts"] = [response_re]
#                 response["create_time"] = time.time()
#                 result["message"] = response
#                 yield json.dumps(result)
#             result["end_turn"] = True
#             yield json.dumps(result, sort_keys=True)
#             yield "[DONE]"
#         else:
#             response_re, _ = chat_model.chat(tokenizer, input, history,
#                                                 max_length=max(chatmodel_onnx["max_tokens"], body.max_tokens),
#                                                 top_p=top_p,
#                                                 temperature=temperature)
#             response["content"]["parts"] = [response_re]
#             response["create_time"] = time.time()
#             result["message"] = response
#             result["end_turn"] = True
#             yield json.dumps(result, sort_keys=True)
#         gc.collect()

#     question_messages = body.messages[-1]
#     if question_messages.role == "user":
#         question = question_messages.content["parts"]
#         id = question_messages.id
#         parent_message_id = question_messages.parent_message_id
#     else:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

#     user_question = ""

#     history = [("你是谁", "我是小欧，你的人工智能助手，请问有什么可以帮到你的？")]
#     completion_text = """The following is a conversation with an AI assistant.
# The assistant is helpful, creative, clever, and very friendly. The assistant is familiar with various languages in the world.

# Human: Hello, who are you?
# AI: I am an AI assistant. How can I help you today?
# Human: 没什么
# AI: 好的, 如果有什么需要, 随时告诉我"""

#     # 判断是否是搜索助手模式
#     # if body.mode == "searchai":
#     #     question_messages = body.messages[-1]
#     #     question_raw = question_messages.content
#     #     question = {}
#     #     question["问题"] = question_raw
#     #     try:
#     #         question["互联网"] = search(question_raw)
#     #     except:
#     #         pass

#     #     history = [("指示:从现在开始，您叫小欧，您将扮演搜索引擎，解答我的问题。我会提供一些互联网结果供你参考，你需要结合自己的知识对相关信息进行提取，但不要编造信息。您可以将回答组织为一个包含不多于5个回答的数字编号列表，每个回答都是与您与用户讨论相关的理解，请注意每回答一个理由后请换行。你要对给定查询进行全面、客观的解答，回答内容不要重复。你可以用数字编号每个回答并在每一条回复的后面提供一些参考的网址。格式如下 [回答](参考的网址) ", "好的，我会总结搜索结果，并按照要求输出。"),
#     # ]

#     # 构造历史输入输出信息
#     for message in body.messages:
#         if message.role == "user":
#             user_question = message.content
#         elif message.role == "system" or message.role == "assistant":
#             assistant_answer = message.content
#             history.append((user_question, assistant_answer))

#     print(history)
#     eval_data = {
#         "conversation": "",
#         "new_message": "",
#         "finished": False
#     }

#     return EventSourceResponse(eval_chatgptlike_chatglm(question, history, id), ping=360)


if __name__ == "__main__":
    uvicorn.run("openai_server:app", host="0.0.0.0",
                port=8005, reload=False, app_dir=".", timeout_keep_alive=180)
