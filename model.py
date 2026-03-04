import torch
import torch.nn as nn

class StockPricePredictor(nn.Module):
    def __init__(self, input_size= 2, hidden_size=[16,8],dropout_rate=0.2):
        super(StockPricePredictor, self).__init__()
        
        layers = []
        in_size=input_size
        for hidden in hidden_size:
            layers.append(nn.Linear(in_size,hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size=hidden
            
        layers.append(nn.Linear(in_size,1))
        self.module=nn.Sequential(*layers)
        
    def forward(self,x):
            return self.module(x)
        
if __name__ == "__main__":
    model = StockPricePredictor(input_size=2, hidden_size=[32 ,16,8], dropout_rate=0.2)
    print(model)
    
    test_input = torch.randn(4,2)
    output = model(test_input)
    print("Output shape:", output.shape)