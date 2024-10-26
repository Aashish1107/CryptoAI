import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, AveragePooling1D
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from boruta import BorutaPy

from sklearn.ensemble import RandomForestClassifier
import warnings
#warnings.filterwarnings('ignore')
print("XX")
from tensorflow.python.keras.layers.recurrent import LSTM
from keras.src.layers import BatchNormalization

class CryptoPredictor:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self.selected_features = None
        self.feature_selector = None
        
    def prepare_features(self, df):
        """Create technical indicators and features"""
        feature_df = df.copy()
        
        # Price-based features
        feature_df['price_change'] = feature_df['close'].pct_change()
        feature_df['price_range'] = (feature_df['high'] - feature_df['low']) / feature_df['open']
        feature_df['price_momentum'] = feature_df['close'].pct_change(3)
        
        # Volume-based features
        feature_df['volume_change'] = feature_df['volume'].pct_change()
        feature_df['volume_price_ratio'] = feature_df['volume'] / feature_df['close']
        feature_df['taker_ratio'] = feature_df['taker_buy_base_volume'] / feature_df['volume']
        
        # Trade-based features
        feature_df['avg_trade_size'] = feature_df['volume'] / feature_df['number_of_trades']
        feature_df['quote_volume_ratio'] = feature_df['quote_asset_volume'] / feature_df['volume']
        
        return feature_df.dropna()
    
    def perform_boruta_selection(self, X, y):
        """Perform Boruta feature selection"""
        rf = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=5
        )
        
        boruta = BorutaPy(
            rf,
            n_estimators='auto',
            verbose=2,
            random_state=42
        )
        
        # Fit Boruta
        boruta.fit(X.values, y.values)
        
        # Get selected features
        selected_features = X.columns[boruta.support_].tolist()
        print("\nSelected Features:", selected_features)
        
        self.feature_selector = boruta
        self.selected_features = selected_features
        
        return selected_features
    
    def prepare_sequences(self, df):
        """Prepare sequences for the model"""
        X, y = [], []
        
        scaled_data = self.scaler.fit_transform(df[self.selected_features + ['target']])
        scaled_df = pd.DataFrame(scaled_data, 
                               columns=self.selected_features + ['target'],
                               index=df.index)
        
        for i in range(self.lookback, len(df)):
            X.append(scaled_df.iloc[i-self.lookback:i][self.selected_features].values)
            y.append(scaled_df.iloc[i]['target'])
            
        return np.array(X), np.array(y)
    
    def build_hybrid_model(self, input_shape):
        """Build the hybrid CNN-LSTM model"""
        model = Sequential([
            # 1D CNN Component
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                  input_shape=input_shape),
            BatchNormalization(),
            AveragePooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            AveragePooling1D(pool_size=2),
            Dropout(0.2),
            
            # LSTM Component
            LSTM(100, activation='tanh', return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(50, activation='tanh'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Final Dense Layers
            Dense(20, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=adam_v2.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def main():
    # Load data
    df = pd.read_csv('./data/train.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Initialize predictor
    predictor = CryptoPredictor(lookback=60)
    
    # Prepare features
    processed_df = predictor.prepare_features(df)
    
    # Perform Boruta feature selection
    features_for_selection = [col for col in processed_df.columns 
                            if col != 'target']
    predictor.perform_boruta_selection(
        processed_df[features_for_selection],
        processed_df['target']
    )
    
    # Prepare sequences
    X, y = predictor.prepare_sequences(processed_df)
    
    # Train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build and train model
    model = predictor.build_hybrid_model((X_train.shape[1], X_train.shape[2]))
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    print("\nF1 Score:", f1_score(y_test, y_pred_binary))

if __name__ == "__main__":
    main()