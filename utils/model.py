from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# import torch


def prepare_model(model_path, provider=None):
    if provider is None:
        provider = "CPUExecutionProvider"
    else:
        pass
    # Load a model from transformers and export it to ONNX
    model = ORTModelForCausalLM.from_pretrained(model_path, provider=provider)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    return model, tokenizer


def stream_response(
    model,
    tokenizer,
    input_text,
    max_output_length=1000,
    output_token_per_iteration=30,
    previous_output_text="",
    end_token=None,
    top_p=0.9,
    temperature=0.6,
):
    if end_token is None:
        end_token = tokenizer.eos_token
    full_text = previous_output_text + input_text
    previous_output_text_len = len(full_text)
    input_ids = tokenizer(full_text, return_tensors="pt")
    for i in range(int(max_output_length / output_token_per_iteration)+1):
        try:
            # 生成文本
            outputs = model.generate(
                **input_ids, max_new_tokens=output_token_per_iteration
            )

            # 判断是否结束
            if end_token in outputs[0]:
                # 找到 end_token 的位置
                end_token_position = outputs[0].tolist().index(end_token)

                # 提取 end_token 之前的内容
                generated_text = tokenizer.decode(outputs[0][:end_token_position])
                output_text = generated_text[previous_output_text_len :]
                yield output_text
                return # 结束

            else:
                # 将生成的文本解码
                generated_text = tokenizer.decode(outputs[0])
                output_text = generated_text[previous_output_text_len :]
                
                # 继续生成文本
                # 将文本转换为模型可接受的输入格式
                input_ids = tokenizer(generated_text, return_tensors="pt")
                yield output_text

        except Exception as e:
            print(f"Error: {e}")
            break


def generate(
    model,
    tokenizer,
    input_text,
    output_length=300,
    previous_output_text="",
    end_token=None,
    top_p=0.9,
    temperature=0.6,
):
    if end_token is None:
        end_token = tokenizer.eos_token
    full_text = previous_output_text + input_text
    previous_output_text_len = len(full_text)
    input_ids = tokenizer(full_text, return_tensors="pt")

    # 生成文本
    outputs = model.generate(
        **input_ids, max_new_tokens=output_length
    )

    # 判断是否结束
    if end_token in outputs[0]:
        # 找到 end_token 的位置
        end_token_position = outputs[0].tolist().index(end_token)

        # 提取 end_token 之前的内容
        generated_text = tokenizer.decode(outputs[0][:end_token_position])
        output_text = generated_text[previous_output_text_len :]
        return output_text


    else:
        # 将生成的文本解码
        generated_text = tokenizer.decode(outputs[0])
        output_text = generated_text[previous_output_text_len :]
        
        # 继续生成文本
        # 将文本转换为模型可接受的输入格式
        input_ids = tokenizer(generated_text, return_tensors="pt")
        return output_text



if __name__ == "__main__":
    import os

    current_dir = os.getcwd()
    parent_dir = current_dir.rsplit("/", 1)[0]
    print(parent_dir)

    model, tokenizer = prepare_model(os.path.join(parent_dir, "qwen1.5-1.8-onnx/"))

    # 结束符号的token id
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    # 最大输出长度
    max_output_length = 300

    previous_output_text = ""
    inputs = ">> User:" + "一句话解释天空为什么是蓝色的？"

    for generated_text in stream_response(
        input_text=inputs,
        max_output_length=max_output_length,
        model=model,
        tokenizer=tokenizer,
        end_token=eos_token_id,
    ):
        print(generated_text)
        previous_output_text = generated_text

    previous_output_text = inputs + previous_output_text
    inputs = ">> User:" + "那火星的呢？"
    for generated_text in stream_response(
        input_text=inputs,
        max_output_length=max_output_length,
        model=model,
        tokenizer=tokenizer,
        previous_output_text=previous_output_text,
        end_token=eos_token_id,
    ):
        print(generated_text)
        previous_output_text = generated_text
