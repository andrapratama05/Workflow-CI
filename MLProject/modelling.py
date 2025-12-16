import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os
import sys
import warnings

if __name__ == '__main__':
    print("Active run:", mlflow.active_run())
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    if mlflow.active_run():
        mlflow.end_run()
    os.environ.pop('MLFLOW_RUN_ID', None)
    
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_dataset_preprocessing.csv")
    
    df = pd.read_csv(file_path)
    X = df.drop(columns=['target'], axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_example = X_train[:5]
    
    pipeline = Pipeline([
        ('rf', RandomForestClassifier())
    ])
    
    mlflow.autolog()
    
    with mlflow.start_run(run_name='Random_Forest_Model'):
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path='model',
            input_example=input_example,
        )
        
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)