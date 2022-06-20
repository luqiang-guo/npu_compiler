import onnx
import torch


def to_onnx(model, dummy_input, onnx_file) :

    # 
    torch.onnx.export(model, dummy_input, onnx_file, opset_version=11)

    onnx_model = onnx.load(onnx_file)

    return onnx_model