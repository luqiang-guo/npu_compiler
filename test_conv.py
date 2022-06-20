import os
import sys
import torch
import copy
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from to_onnx import to_onnx
from util import accuracy

# 
NPU_ROOTPATH = "/home/guoluqiang/nova/compiler/"
# quantizer
sys.path.append(NPU_ROOTPATH + "npu_quantizer")
sys.path.append(NPU_ROOTPATH + "npu_compiler") 

from quantize import run_quantizer
from compile import run_compiler

onnx_filename = "test_conv.onnx"
model_filename = "./models/" + "test_conv" +".pth"
test_shape = (1, 6, 6)

class TestDataset(Dataset):
    def __init__(self, len):
        self.len = len
        self.images = []
        self.labels = []

        for i in range(len):
            self.images.append(torch.ones(test_shape, dtype=torch.int8) * i / 255.0)
            self.labels.append(torch.tensor(i) * 1.0)
 
    def __len__(self):
        return  self.len
 
    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def get_dataloader():

    train_dataset = TestDataset(8)
    val_dataset = TestDataset(8)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
    )
    return train_loader, val_loader

class TestConv (nn.Module):
    def __init__(self):
        super(TestConv, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3, stride=3)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        y = self.conv(x)
        z = self.relu(y)
        return self.max_pool(z).view(-1)

def train(batch_size, epochs, use_gpu):
    print("start -->")
    
    # dataloader
    train_loader, val_loader = get_dataloader()

    # model
    model = TestConv()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_gpu:
        model = model.to(device)

    # optimizer
    learning_rate = 0.0005
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate , momentum=momentum)

    #
    criterion = nn.MSELoss().to(device)

    # train
    model.train()

    for epoch in range(epochs):
        # train
        for input, target in train_loader:
            #
            if use_gpu:
                input = input.to(device)
                target = target.to(device)
            output = model.forward(input)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss
            # print("loss : ", loss)

        
    # save models
    torch.save(model, model_filename)
    print("save ", model_filename)

def test(model):
    x = (torch.ones(test_shape, dtype=torch.int8) * 2 / 255.0).reshape(1,1,6,6)
    print(x.size())
    y = model(x)
    print(y)

def main():

    if(os.path.exists(model_filename) != True):
        train(1, 50, True)

    model = torch.load(model_filename).to("cpu")
    model.eval()

    # test
    test(model)

    # dummy_input = torch.randn((1,1,6,6), requires_grad=True)
    # y = model(dummy_input)

    # onnx_model = to_onnx(model, dummy_input, onnx_filename)
    # print(onnx_model)

    # _, val_loader = get_dataloader()
    # ir_graph = run_quantizer(onnx_model, [(1,1,6,6)], dataloader=val_loader,
    #                             num_batchs=1, save_dir='./ir_output', debug=True, load_type="onnx")

    # run_compiler(input_dir='./ir_output', output_dir='./compiler_output',
    #              enable_cmodel=True, enable_rtl_model=True, enable_profiler=True)

if __name__ == "__main__" :
    main()