from tinygrad.tensor import Tensor
import numpy as np

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.filters = Tensor.uniform(out_channels, in_channels, kernel_size, kernel_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size
        out_H = H - K + 1
        out_W = W - K + 1
        out = Tensor.zeros(B, self.out_channels, out_H, out_W)
        
        x_data = x.data()
        filters_data = self.filters.data()
        out_data = out.data()
        
        for b in range(B):
            for f in range(self.out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        total = 0.0
                        for c in range(self.in_channels):
                            for ki in range(K):
                                for kj in range(K):
                                    x_val = x_data[b, c, i + ki, j + kj]
                                    w_val = filters_data[f, c, ki, kj]
                                    total += x_val * w_val
                        out_data[b, f, i, j] = total
        
        for b in range(B):
            for f in range(self.out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        if out_data[b, f, i, j] < 0:
                            out_data[b, f, i, j] = 0.0
        
        return out


if __name__ == "__main__":
    np.random.seed(0)
    
    input_data = np.array([[[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]]], dtype=np.float32)
    input_tensor = Tensor(input_data)
    
    print("Input Tensor:")
    print(input_tensor.data())  
    conv = ConvLayer(in_channels=1, out_channels=1, kernel_size=3)
    
    print("\nConvolution Filters (Weights):")
    print(conv.filters.data())  
    
    # Forward pass
    output = conv.forward(input_tensor)
    
    print("\nOutput Tensor after Convolution and ReLU:")
    print(output.data())