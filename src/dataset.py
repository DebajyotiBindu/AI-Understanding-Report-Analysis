import pandas as pd
import os

DATA_PATH=r"D:\mlproject15\data\mtsamples.csv\mtsamples.csv"

def dataset(path=DATA_PATH):
    if os.path.exists(path):
        data=pd.read_csv(path)
    else:
        raise FileNotFoundError(f"Dataset not found at {path}")

    data.dropna(subset=["medical_specialty","transcription"],inplace=True)
    data=data.sample(frac=1,random_state=42).reset_index(drop=True)

    splits=["train","val","test"]
    for split in splits:
        if split=='train':
            os.makedirs("data/train",exist_ok=True)
            train_path=os.path.join("data","train",f"{split}.csv")
            train_data=data[:int(0.8*len(data))]
            train_data.to_csv(train_path,index=False)
        
        elif split=='val':
            os.makedirs("data/val",exist_ok=True)
            val_path=os.path.join("data","val",f"{split}.csv")
            val_data=data[int(0.8*len(data)):int(0.9*len(data))]
            val_data.to_csv(val_path,index=False)
        
        else:
            os.makedirs("data/test",exist_ok=True)
            test_path=os.path.join("data","test",f"{split}.csv")
            test_data=data[int(0.9*len(data)):]
            test_data.to_csv(test_path,index=False)

    print("Dataset created successfully")  
    return 

if __name__=="__main__":
    dataset()