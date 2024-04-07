import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import numpy as np
import datetime as dt

# read in data
data = pd.read_csv('motor_data.csv')
# START OF PREPROCESS DATA
label = preprocessing.LabelEncoder()
gender = label.fit_transform(data['SEX'])
data['SEX'] = gender
data['INSR_BEGIN'] = pd.to_datetime(data['INSR_BEGIN'], format='%d-%b-%y')
data['INSR_BEGIN'] = data['INSR_BEGIN'].map(dt.datetime.toordinal)
data['INSR_END'] = pd.to_datetime(data['INSR_END'], format='%d-%b-%y')
data['INSR_END'] = data['INSR_END'].map(dt.datetime.toordinal)
carr = label.fit_transform(data['CARRYING_CAPACITY'])
data['CARRYING_CAPACITY'] = carr
vtype = label.fit_transform(data['TYPE_VEHICLE'])
data['TYPE_VEHICLE'] = vtype
vMake = label.fit_transform(data['MAKE'])
data['MAKE'] = vMake
vUsage = label.fit_transform(data['USAGE'])
data['USAGE'] = vUsage
vEffectiveYr = label.fit_transform(data['EFFECTIVE_YR'])
data['EFFECTIVE_YR'] = vEffectiveYr
# END OF PREPROCESS DATA

# Remove unnecessary columns from data
data = data.drop('OBJECT_ID', axis=1)
data = data.drop('CARRYING_CAPACITY', axis=1)
data = data.drop('CLAIM_PAID', axis=1)
data = data.drop('CCM_TON', axis=1)
data = data.drop('EFFECTIVE_YR', axis=1)
data = data.drop('INSR_TYPE', axis=1)
df = pd.DataFrame(data)


def filter(premium):
    if premium <= 500:
        return 0
    elif premium <= 1000:
        return 1
    elif premium <= 2000:
        return 2
    elif premium <= 5000:
        return 3
    elif premium <= 10000:
        return 4
    elif premium <= 20000:
        return 5
    elif premium <= 50000:
        return 6
    elif premium <= 100000:
        return 7
    elif premium <= 200000:
        return 8
    elif premium > 200000:
        return 9


df = df.dropna(axis=0, subset=None, inplace=False, ignore_index=False)
df['CATEGORY'] = (df['PREMIUM'].apply(filter))
target = (df['CATEGORY'])

# Normalize the data
normalized_df = df.copy()
normalized_df = (df-df.min())/(df.max()-df.min())

# split data randomly so 70% is training and 30% is testing
normalized_df = normalized_df.drop('PREMIUM', axis=1)
normalized_df = normalized_df.drop('CATEGORY', axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(
    normalized_df, target, test_size=.3, random_state=42)

#CREATE SEQUENTIAL NEURAL NETWORK
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

X_train = tf.constant((X_train))
Y_train = tf.constant((Y_train))
X_test = tf.constant((X_test))
Y_test = tf.constant((Y_test))
history = model.fit(x=X_train, y=Y_train, batch_size=32,
                    epochs=13, shuffle=True,
                    validation_split=0.1)
model.evaluate(x=X_test, y=Y_test, batch_size=32)

model.save('saved_models/seq1.keras')
