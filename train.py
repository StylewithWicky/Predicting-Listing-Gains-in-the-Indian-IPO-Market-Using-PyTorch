import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import feature_enginnering
from model import StockPricePredictor

def coach():
    train_ds, test_ds, = feature_enginnering("data_processed/ipo_tracker_cleaned.csv")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    model = StockPricePredictor(input_size=2, hidden_size=[32,16,8], dropout_rate=0.2)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 1000
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "models/stock_price_predictor.pth")
    print("Model saved to models/stock_price_predictor.pth")
    
if __name__ == "__main__":
    coach()
    