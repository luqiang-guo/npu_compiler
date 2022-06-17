import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import onnx
import sys


# 
NPU_ROOTPATH = "/home/guoluqiang/nova/compiler/"
# quantizer
sys.path.append(NPU_ROOTPATH + "npu_quantizer") 

from quantize import run_quantizer

input_shape = (1)
onnx_file = "test_relu.onnx"

class ReluDataset(Dataset):
    def __init__(self, len):
        self.len = len
 
    def __len__(self):
        return  self.len
 
    def __getitem__(self, index):
        return torch.randn(input_shape), torch.randn(input_shape)


def get_dataloader():

    val_dataset = ReluDataset(16)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
    )
    return val_loader


class TestRelu (nn.Module):
    def __init__(self):
        super(TestRelu, self).__init__()

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(x)
        return y

def to_onnx(model) :
    model.eval()

    dummy_input = torch.randn(input_shape, requires_grad=True)

    torch.onnx.export(model, dummy_input, onnx_file, opset_version=11)

    onnx_model = onnx.load(onnx_file)

    return onnx_model

def main():

    test_relu = TestRelu()
    x = torch.randn(input_shape)
    y = test_relu(x)

    print(y)

    # to onnx
    onnx_model = to_onnx(test_relu)
    print(onnx_model)

    val_loader = get_dataloader()

    ir_graph = run_quantizer(onnx_model, [input_shape], dataloader=val_loader,
                         num_batchs=1, save_dir='./ir_output', debug=True, load_type="onnx")


if __name__ == "__main__":
    main()
