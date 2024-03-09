from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B-Chat")
    parser.add_argument("--save_dir", type=str, default="models/qwen1.5-7-Chat")
    parser.add_argument("--onnx", type=str, default="", help="Export pytorch to ONNX format.")
    parser.add_argument(
        "--quantize",
        type=str,
        default="avx2",
        choices=["avx2", "avx512", "avx512_vnni"],
    )
    args = parser.parse_args()
    save_directory = parser.parse_args().save_dir
    if bool(args.onnx) == True:
        # Load a model from transformers and export it to ONNX
        ort_model = ORTModelForCausalLM.from_pretrained(
            args.model, export=True, task="text-generation-with-past"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        # Save the onnx model and tokenizer

        ort_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
    

    
    if args.quantize == "avx2":
        dqconfig = AutoQuantizationConfig.avx2(
            is_static=False,
            per_channel=False,
            use_symmetric_activations=True,
            operators_to_quantize=["MatMul"],
        )
    elif args.quantize == "avx512":
        dqconfig = AutoQuantizationConfig.avx512(
            is_static=False,
            per_channel=False,
            use_symmetric_activations=True,
            operators_to_quantize=["MatMul"],
        )
    elif args.quantize == "avx512_vnni":
        dqconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False,
            use_symmetric_activations=True,
            operators_to_quantize=["MatMul"],
        )

    quantizer = ORTQuantizer.from_pretrained(save_directory)

    model_quantized_path = quantizer.quantize(
        save_dir=save_directory + "-" + args.quantize + "-quantizer",
        quantization_config=dqconfig,
        use_external_data_format=True
    )
