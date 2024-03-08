import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "D:\lzy\Documents\workspace\llm\qwen1.5-1.8-onnx\model.onnx"
model_quant = "D:\lzy\Documents\workspace\llm\qwen1.5-1.8-onnx\model_int8.onnx"
quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QInt8)
