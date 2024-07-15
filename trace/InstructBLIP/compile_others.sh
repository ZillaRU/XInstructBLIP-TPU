# ./compile_llm.sh --mode int8 --name llama2-7b # same as int4

#!/bin/bash
source /workspace/tpu-mlir/envsetup.sh

model_transform.py \
    --model_name Qformer \
    --model_def ./onnx/Qformer/Qformer_onnx.onnx \
    --mlir Qformer.mlir \
    --test_input ./fake_inputs.npz \
    --test_result ./debug/qformer_out_top.npz

model_deploy.py \
    --mlir Qformer.mlir \
    --quantize F32 \
    --chip bm1684x \
    --model Qformer_F32.bmodel \
    --test_input ./fake_inputs.npz \
    --test_reference ./debug/qformer_out_top.npz \
    --compare_all

# model_transform.py \
#     --model_name visual_encoder.pt \
#     --model_def ./onnx/visual_encoder/visual_encoder_onnx.onnx \
#     --mlir visual_encoder.mlir

# model_deploy.py \
#     --mlir visual_encoder.mlir \
#     --quantize F32 \
#     --chip bm1684x \
#     --model visual_encoder_F32.bmodel