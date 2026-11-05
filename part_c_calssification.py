import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

from src.shipment_prep import FEATURES,NOISE_COLUMNS

df=pd.read_csv("data/modelling_table.csv")
X=df[FEATURES]
y=df["is_delayed"]

# splitting data
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

X_test,X_val