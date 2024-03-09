from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import argparse
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime import ORTOptimizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--save_dir", type=str, default="models/qwen1.5-0.5-Chat")
    parser.add_argument("--onnx", type=str, default="true", help="Export pytorch to ONNX format.", choices=["true", "false"])
    parser.add_argument(
        "--quantize",
        type=str,
        default="avx2",
        choices=["none", "avx2", "avx512", "avx512_vnni"],
    )
    parser.add_argument("--quantized_op", type=list, default=["MutMul"])
    parser.add_argument("--opt", type=int, choices=[0, 1, 2, 99], default=0, help="Optimization level.")
    args = parser.parse_args()
    save_directory = parser.parse_args().save_dir
    
    
    if  args.onnx == "true":
        # Load a model from transformers and export it to ONNX
        ort_model = ORTModelForCausalLM.from_pretrained(
            args.model, export=True, task="text-generation-with-past"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        
        
    else:
        # Load a model from transformers and export it to ONNX
        ort_model = ORTModelForCausalLM.from_pretrained(
            save_directory, task="text-generation-with-past"
        )
        tokenizer = AutoTokenizer.from_pretrained(save_directory)
    
    if args.opt != 0:
        print("Optimizing the model...")
        optimizer = ORTOptimizer.from_pretrained(ort_model)
        optimization_config = OptimizationConfig(optimization_level=args.opt)
        print("Exporting the OPT model to ONNX...")
        optimizer.optimize(save_dir=save_directory+"-opt", optimization_config=optimization_config)
    elif args.onnx == "true":
        print("Optimization is not enabled.")
        ort_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        

    
    if args.quantize == "avx2":
        dqconfig = AutoQuantizationConfig.avx2(
            is_static=False,
            per_channel=False,
            use_symmetric_activations=True,
            operators_to_quantize=args.quantized_op,
        )
    elif args.quantize == "avx512":
        dqconfig = AutoQuantizationConfig.avx512(
            is_static=False,
            per_channel=False,
            use_symmetric_activations=True,
            operators_to_quantize=args.quantized_op,
        )
    elif args.quantize == "avx512_vnni":
        dqconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False,
            use_symmetric_activations=True,
            operators_to_quantize=args.quantized_op,
        )

    if args.quantize != "none":
        print("Quantizing the model...", args.quantize)
        quantizer = ORTQuantizer.from_pretrained(ort_model)

        model_quantized_path = quantizer.quantize(
            save_dir=save_directory + "-" + args.quantize + "-quantizer",
            quantization_config=dqconfig,
            use_external_data_format=True
        )