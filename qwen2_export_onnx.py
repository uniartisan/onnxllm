from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


save_directory = "qwen1.5-7-onnx/"
# Load a model from transformers and export it to ONNX
ort_model = ORTModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat", export=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
# Save the onnx model and tokenizer

ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

quantizer = ORTQuantizer.from_pretrained(save_directory)


dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
model_quantized_path = quantizer.quantize(
    save_dir=save_directory+"-avx512-quantizer",
    quantization_config=dqconfig,
)