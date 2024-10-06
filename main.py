from tinygrad.tensor import Tensor
import numpy as np

class LowLevelNN:
    def __init__(self):

        self.input_to_hidden = Tensor.uniform(2, 128)  
        self.hidden_to_output = Tensor.uniform(128, 1) 

    def forward(self, x):
        x_data = x.data()
        w1_data = self.input_to_hidden.data()
        w2_data = self.hidden_to_output.data()

        batch_size = x_data.shape[0]
        input_dim = x_data.shape[1]
        hidden_dim = w1_data.shape[1]
        output_dim = w2_data.shape[1]
        hidden_activation = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        output_activation = np.zeros((batch_size, output_dim), dtype=np.float32)

        for i in range(batch_size):
            for h in range(hidden_dim):
                total = 0.0
                for j in range(input_dim):
                    total += x_data[i, j] * w1_data[j, h]
                hidden_activation[i, h] = total
        for i in range(batch_size):
            for h in range(hidden_dim):
                if hidden_activation[i, h] < 0:
                    hidden_activation[i, h] = 0.0
        for i in range(batch_size):
            for o in range(output_dim):
                total = 0.0
                for h in range(hidden_dim):
                    total += hidden_activation[i, h] * w2_data[h, o]
                output_activation[i, o] = total
        for i in range(batch_size):
            for o in range(output_dim):
                output_activation[i, o] = 1 / (1 + np.exp(-output_activation[i, o]))
        self.hidden_activation = Tensor(hidden_activation) 
        self.output_activation = Tensor(output_activation)

        return self.output_activation

    def backward(self, loss_grad):
        self.output_activation.backward(loss_grad)

    def update(self, lr=0.01):
        if self.input_to_hidden.grad is not None:
            self.input_to_hidden -= self.input_to_hidden.grad * lr
            self.input_to_hidden.grad = None  
        if self.hidden_to_output.grad is not None:
            self.hidden_to_output -= self.hidden_to_output.grad * lr
            self.hidden_to_output.grad = None
