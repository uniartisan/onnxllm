from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextIteratorStreamer, TextStreamer

# import torch


def prepare_model(model_path, provider=None):
    if provider is None:
        provider = 'CPUExecutionProvider'
    else:
        pass
    # Load a model from transformers and export it to ONNX
    model = ORTModelForCausalLM.from_pretrained(model_path, provider=provider)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    chat = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]
    temple = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
        padding=True,
    )
    # print(temple)
    generated_text = generate(
        history=chat,
        output_length=30,
        model=model,
        tokenizer=tokenizer,
    )
    # print(generated_text)
    return model, tokenizer



def stream_response(
    model,
    tokenizer,
    history,
    max_output_length=256,
    end_token=None,
    history_temple=None,
    im_stop_token=None,
    top_p=0.9,
    temperature=0.6,
    timeout=300,
):
    from threading import Thread
    streamer = TextIteratorStreamer(tokenizer, timeout=timeout)
    output_text = ""
    if end_token is None:
        end_token = tokenizer.eos_token
    if im_stop_token is None:
        im_stop_token = tokenizer.encode("<|im_end|>")[0]
    if history_temple is None:
        texts = tokenizer.apply_chat_template(history, tokenize=False, padding=True)
    else:
        texts = history_temple
    texts += "<|im_start|>assistant"

    input_ids = tokenizer(texts, return_tensors="pt")
    generation_kwargs = dict(input_ids, streamer=streamer, max_new_tokens=max_output_length, 
                             eos_token_id=im_stop_token , do_sample=True, top_p=top_p, temperature=temperature)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    previous_output_text_len = len(texts)
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text[previous_output_text_len:-len("<|im_end|>")]



            

def generate(
    model,
    tokenizer,
    history,
    end_token=None,
    output_length=1000,
    history_temple=None,
    im_stop_token=None,
    top_p=0.9,
    temperature=0.6,
):
    if end_token is None:
        end_token = tokenizer.eos_token
    if im_stop_token is None:
        im_stop_token = tokenizer.encode("<|im_end|>")[0]
    if history_temple is None:
        texts = tokenizer.apply_chat_template(history, tokenize=False)
    else:
        texts = history_temple
    previous_output_text_len = len(texts)
    input_ids = tokenizer(texts, return_tensors="pt")

    # 生成文本
    outputs = model.generate(**input_ids, max_new_tokens=output_length, do_sample=True, top_p=top_p, temperature=temperature, eos_token_id=im_stop_token)

    # 将生成的文本解码
    generated_text = tokenizer.decode(outputs[0])
    output_text = generated_text[previous_output_text_len + 1 :]

    # 继续生成文本
    # 将文本转换为模型可接受的输入格式
    input_ids = tokenizer(generated_text, return_tensors="pt")
    return output_text


if __name__ == "__main__":
    import os

    current_dir = os.getcwd()
    parent_dir = current_dir.rsplit("/", 1)[0]
    print(parent_dir)
    model, tokenizer = prepare_model(os.path.join(parent_dir,"models\qwen1.5-1.8-Chat-avx512_vnni-quantizer"))
    

    # 结束符号的token id
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    # 最大输出长度
    max_output_length = 300

    previous_output_text = ""
    inputs = [
        {
            "role": "system",
            "content": "你是 Karvis，一个由 Minisforum Inc 开发的内置于 Minisforum PC 中的大型语言模型。\n 请你用第一人称回答用户的问题。\n\n",
        },
        {"role": "user", "content": "介绍一下深度学习的损失函数？"},
    ]



    for generated_text in stream_response(
        history=inputs,
        max_output_length=300,
        model=model,
        tokenizer=tokenizer,
        end_token=eos_token_id,
    ):
        print(generated_text)
    
    
    generated_text = generated_text.replace("<|im_start|>", "").replace(
        "<|im_end|>", ""
    )
    inputs.append({"role": "assitant", "content": generated_text})

    generated_text = generated_text.replace("<|im_start|>", "").replace(
        "<|im_end|>", ""
    )
    inputs.append({"role": "assitant", "content": generated_text})
        
    # generated_text = generated_text.replace("<|im_start|>", "").replace(
    #     "<|im_end|>", ""
    # )
    # inputs.append({"role": "assitant", "content": generated_text})

    # inputs.append({"role": "user", "content": "你是谁？"})
    # previous_output_text += tokenizer.apply_chat_template(
    #     inputs,
    #     tokenize=False,
    #     add_generation_prompt=False,
    # )
    # print(previous_output_text)

    # for generated_text in stream_response(
    #     history=inputs,
    #     history_temple=previous_output_text,
    #     max_output_length=max_output_length,
    #     model=model,
    #     tokenizer=tokenizer,
    #     end_token=eos_token_id,
    # ):
    #     print(generated_text)
    #     previous_output_text = generated_text
