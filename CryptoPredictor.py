import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from boruta import BorutaPy

from sklearn.ensemble import RandomForestClassifier
import warnings
#warnings.filterwarnings('ignore')

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization

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
        #feature_df['price_change'] = feature_df['close'].pct_change()
        #feature_df['price_range'] = (feature_df['high'] - feature_df['low']) / feature_df['open']
        #feature_df['price_momentum'] = feature_df['close'].pct_change(3)

        # Volume-based features
        #feature_df['volume_change'] = feature_df['volume'].pct_change()
        #feature_df['volume_price_ratio'] = feature_df['volume'] / feature_df['close']
        #feature_df['taker_ratio'] = feature_df['taker_buy_base_volume'] / feature_df['volume']

        # Trade-based features
        #feature_df['avg_trade_size'] = feature_df['volume'] / feature_df['number_of_trades']
        #feature_df['quote_volume_ratio'] = feature_df['quote_asset_volume'] / feature_df['volume']

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
        boruta.fit(np.array(X.values), np.array(y.values))

        # Get selected features
        selected_features = X.columns[boruta.support_].tolist()
        print("\nSelected Features:", selected_features)

        self.feature_selector = boruta
        self.selected_features = selected_features

        return selected_features

    def prepare_sequences(self, df, stage='train'):
        """Prepare sequences for the model"""
        X, y = [], []
        if stage=='train':
            scaled_data = self.scaler.fit_transform(df[self.selected_features + ['target']])
            scaled_df = pd.DataFrame(scaled_data,
                                columns=self.selected_features + ['target'],
                                index=df.index)

            for i in range(self.lookback, len(df)):
                X.append(scaled_df.iloc[i-self.lookback:i][self.selected_features].values)
                y.append(scaled_df.iloc[i]['target'])

            return np.array(X), np.array(y)
        else:
            for i in range(self.lookback, len(df)):
                X.append(df.iloc[i-self.lookback:i][self.selected_features].values)
            return np.array(X)

    def build_hybrid_model(self, input_shape):
        """Build the hybrid CNN-LSTM model"""
        model = Sequential([
            # 1D CNN Component
            #BatchNormalization(input_shape=input_shape),
            #Conv1D(filters=64, kernel_size=3, padding='same' , activation='relu'),
            #BatchNormalization(),
            #Dropout(0.1),

            Conv1D(filters=64, kernel_size=3,activation='relu',input_shape=input_shape),
            BatchNormalization(),
            AveragePooling1D(pool_size=1),
            Dropout(0.1),

            # LSTM Component
            LSTM(128, activation='tanh', return_sequences=True),
            BatchNormalization(),
            Dropout(0.1),

            LSTM(80, activation='tanh'),
            BatchNormalization(),
            Dropout(0.1),

            # Final Dense Layers
            #Dense(20, activation='relu'),
            #BatchNormalization(),
            #Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(
                learning_rate=0.001,
            ),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]
        )

        return model

def main():
    # Load data
     # Initialize predictor
    predictor = CryptoPredictor(lookback=5)
    processed_df = pd.read_csv('./data/processed_df.csv')
    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], unit='s')
    processed_df.set_index('timestamp', inplace=True)
    predictor.selected_features = processed_df.columns.tolist()
    predictor.selected_features.remove('target')
    print(processed_df)
    print(predictor.selected_features)
    print("Start Training")
    
    

    #Sequences
    print(".................Preparing Features")
    # Prepare features
    print("...Preparing Sequences")
    #print(processed_df)
    # Prepare sequences
    l=processed_df.shape[0]
    #l//1000
    for i in range(1,l//1000):
        if i!=1 and i%100!=0:
            continue
        print("Sequence "+str(i))
        if i==1:
            X_train, y_train = predictor.prepare_sequences(processed_df.iloc[(i-1)*1000:i*1000])
        else:
            X_train, y_train = predictor.prepare_sequences(processed_df.iloc[(i-1)*1000-predictor.lookback:i*1000])
        print("...Passing to model")
        print(X_train.shape)
        print(y_train.shape)
        #print(y_train)
        # Train model
        if i==1:
            model = predictor.build_hybrid_model((X_train.shape[1], X_train.shape[2]))
            # Setup callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
                ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),
            ]
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.01,
            callbacks=callbacks,
            verbose=1
        )
        if i%10==0:
            model.save_weights('./Data/Weights.weights.h5')
            print("Saving Weights at Sequence "+str(i))
    print("Last Sequence")
    X_train, y_train = predictor.prepare_sequences(processed_df.iloc[(l//1000)*1000-predictor.lookback:l])
    print(X_train.shape)
    print(y_train.shape)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.01,
        callbacks=callbacks,
        verbose=1
    )
    print("Training Over")
    model.save_weights('./Data/Weights.weights.h5')
    print("SAVED WEIGHTS")
    exit()

    

def test():
    print("...Validation Check")
    # Evaluate model
    predictor = CryptoPredictor(lookback=5)
    processed_df = pd.read_csv('./data/processed_df.csv')
    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], unit='s')
    processed_df.set_index('timestamp', inplace=True)
    processed_df=predictor.prepare_features(processed_df)
    
    test_df = pd.read_csv('./data/test.csv')
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], unit='s')
    test_df.set_index('timestamp', inplace=True)
    predictor.selected_features = test_df.columns.tolist()
    predictor.selected_features.remove('row_id')
    processed_df=processed_df[predictor.selected_features]
    test_df=test_df[predictor.selected_features]
    processed_df=processed_df.tail(5)
    #print(processed_df)
    #print(test_df)
    test_df=pd.concat([processed_df,test_df], axis=0)
    print(test_df)
    
    X_test = predictor.prepare_sequences(test_df, 'test')
    print(X_test.shape)
    
    model = predictor.build_hybrid_model((X_test.shape[1], X_test.shape[2]))
    model.load_weights('./Data/Weights.weights.h5')
    #print(X_test)
    #print(X_test.shape)
    
    y_pred=model.predict(X_test)
    y_pred_binary = list((y_pred > 0.5).astype(int).flatten())
    print("No. of 1s="+ str(sum(y_pred_binary)))
    print("No. of 0s="+str(len(y_pred_binary)-sum(y_pred_binary)))
    row_id=[i for i in range(len(y_pred_binary))]
    test_df=pd.DataFrame({'row_id': row_id, 'target': y_pred_binary})
    test_df.set_index('row_id', inplace=True)
    print(test_df)
    test_df.to_csv('./Data/Submissions.csv')
    print('Uploaded')
if __name__ == "__main__":
    test()