import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class Preprocessor(Dataset):
    
    def __init__(self,X,Y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.Y=torch.tensor(Y,dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]
    
def feature_enginnering(file_path):
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df =pd.read_csv(file_path)
        print(f"Loaded data with {len(df)} entries and {len(df.columns)} columns.")
        
        df['Subscription (x)']=pd.to_numeric(df['Subscription (x)'],errors='coerce')
        df['Subscription (x)'] =df["Subscription (x)"].fillna(df["Subscription (x)"].median())
        feature=['Issue Price (Rs.)','Subscription (x)']
        target=['Listing_Gain_Pct']
        
        df=df.dropna(subset=feature+target)
        
        X=df[feature].values
        Y=df[target].values.reshape(-1,1)
        
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
        
        scaler_X=StandardScaler()
        X_train=scaler_X.fit_transform(X_train)
        X_test=scaler_X.transform(X_test)
        
        scaler_Y=StandardScaler()
        Y_train=scaler_Y.fit_transform(Y_train)
        Y_test=scaler_Y.transform(Y_test)
        
        os.makedirs("models",exist_ok=True)
        joblib.dump(scaler_X,"models/scaler_X.pkl")
        joblib.dump(scaler_Y,"models/scaler_Y.pkl")
        
        return Preprocessor(X_train,Y_train),Preprocessor(X_test,Y_test)
        
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return None, None
if __name__ == "__main__":
    
    train_ds, test_ds = feature_enginnering("data_processed/ipo_tracker_cleaned.csv")
    
    if train_ds is not None:
        
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

        
        sample_features, sample_targets = next(iter(train_loader))

        print("\n" + "="*30)
        print(" FINAL PREPROCESSING REPORT")
        print("="*30)
        print(f" Training Samples: {len(train_ds)}")
        print(f" Testing Samples:  {len(test_ds)}")
        print(f" Feature Shape:    {sample_features.shape}") 
        print(f" Target Shape:     {sample_targets.shape}")
        
       
        print(f"\n Scaled Feature Sample: {sample_features[0].tolist()}")
        print(f"  Scaled Target Sample:  {sample_targets[0].tolist()}")
        
        if abs(sample_features.mean()) < 1.0:
            print("\nSTATUS: Ready for Brain Construction.")
        print("="*30)