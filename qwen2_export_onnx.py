from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer


save_directory = "qwen1.5-7-onnx/"
# Load a model from transformers and export it to ONNX
ort_model = ORTModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat", export=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
# Save the onnx model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)