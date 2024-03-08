from optimum.onnxruntime import ORTModelForCausalLM
from transformers import pipeline, AutoTokenizer

# import torch


def prepare_model(model_path, provider=None):
    if provider is None:
        provider = "CPUExecutionProvider"
    else:
        pass
    # Load a model from transformers and export it to ONNX
    model = ORTModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    return model, tokenizer


def stream_response(
    input_text,
    max_output_length,
    model,
    tokenizer,
    previous_output_text=None,
    end_token=None,
):
    if end_token is None:
        end_token = tokenizer.eos_token

    input_ids = tokenizer(previous_output_text + input_text, return_tensors="pt")
    while True:
        try:
            # 生成文本
            outputs = model.generate(**input_ids, max_new_tokens=15)

            # 判断是否结束
            if end_token in outputs[0]:
                # 找到 end_token 的位置
                end_token_position = outputs[0].tolist().index(end_token)

                # 提取 end_token 之前的内容
                output_text = tokenizer.decode(outputs[0][:end_token_position])
                return output_text[len(previous_output_text) :]

            else:
                # 将生成的文本解码
                generated_text = tokenizer.decode(outputs[0])

                # 将文本转换为模型可接受的输入格式
                input_ids = tokenizer(generated_text, return_tensors="pt")

                # 避免无限循环，可以设置最大生成token数
                if len(input_text) >= max_output_length:
                    return generated_text[len(previous_output_text) :]

                # 继续生成文本
                yield generated_text[len(previous_output_text) :]

        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    model, tokenizer = prepare_model("qwen1.5-1.8-onnx")

    input_text = ">> User:" + "天空为什么是蓝色的？"
    end_token = 151643
    # 结束符号的token id
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    # 最大输出长度
    max_output_length = 1500
    prompt = input_text
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt")
    # Warm up the model
    while True:
        # 生成文本
        outputs = model.generate(**input_ids, max_new_tokens=15)

        # 判断是否结束
        if end_token in outputs[0]:
            # 找到 end_token 的位置
            end_token_position = outputs[0].tolist().index(end_token)

            # 提取 end_token 之前的内容
            output_text = tokenizer.decode(outputs[0][:end_token_position])
            print(output_text)
            break
        else:
            # 将生成的文本解码
            generated_text = tokenizer.decode(outputs[0])
            print(generated_text)

        # 将文本转换为模型可接受的输入格式
        input_ids = tokenizer(generated_text, return_tensors="pt")

        # 避免无限循环，可以设置最大生成token数
        if len(input_text) >= max_output_length:
            break

    generated_text += ">> User:" + "还有可能是其他的颜色吗？"
    input_ids = tokenizer(generated_text, return_tensors="pt")
    while True:
        # 生成文本
        outputs = model.generate(**input_ids, max_new_tokens=15)

        # 判断是否结束
        if end_token in outputs[0]:
            # 找到 end_token 的位置
            end_token_position = outputs[0].tolist().index(end_token)

            # 提取 end_token 之前的内容
            output_text = tokenizer.decode(outputs[0][:end_token_position])
            print(output_text)
            break
        else:
            # 将生成的文本解码
            generated_text = tokenizer.decode(outputs[0])
            print(generated_text)

        # 将文本转换为模型可接受的输入格式
        input_ids = tokenizer(generated_text, return_tensors="pt")

        # 避免无限循环，可以设置最大生成token数
        if len(input_text) >= max_output_length:
            break

    # input_ids = tokenizer(input_text, return_tensors="pt")
    # outputs = model.generate(**input_ids)
