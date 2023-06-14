import datetime
import os
import shutil
import numpy as np
import tensorflow as tf
import hypertune
import numpy as np
from sklearn import preprocessing
from google.cloud import bigquery, storage
from google.oauth2 import credentials
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Softmax)

# Define the CKD-EPI equation function
def calculate_eGFR(row):
    if row['Sex'] == 'male':
        kappa = 0.9
        alpha = -0.302
        beta = 1.0
    else:
        kappa = 0.7
        alpha = -0.241
        beta = 1.012

    eGFR = 142 * min(row['serum_creatinine'] / kappa, 1)**alpha * \
           max(row['serum_creatinine'] / kappa, 1)**(-1.2) * \
           0.9938**row['Patient.Age.at.Biopsy'] * beta
    return eGFR

def get_add_var(image_input_dir):
    df = pd.read_csv(image_input_dir)
    df['eGFR'] = df.apply(calculate_eGFR, axis=1)
    df['time.TX']=abs(df['Patient.Age.at.Biopsy'] - df['Patient.Age.at.TX'])
    eGFR_bins = [float('-inf'), 60, 89, float('inf')]
    TimeTX_bins = [float('-inf'), 1, float('inf')]

    # Create the binned columns for 'eGFR' and 'Time.TX'
    df['eGFR_bin'] = pd.cut(df['eGFR'], bins=eGFR_bins, labels=['<60', '60-89', '>=90'])
    df['time.TX_bin'] = pd.cut(df['time.TX'], bins=TimeTX_bins, labels=['<1 year', '>1 year'])
    
    return df

CSV_COLUMNS = [
    "serum_creatinine",
    "urea",
    "dimethylamine",
    "UA.Pro",
    "phenylacetylglutamine",
    "Hypertension",
    "trigonellin",
    "lactate",
    "citrate",
    "hippurate",
    "Sex",
    "alanine",
    "Diabetes",
    "UA.Hb",
    "eGFR",
    "time.TX",
    "eGFR_bin",
    "time.TX_bin",
    "Case"
]
LABEL_COLUMN = "Case"

NUMERICAL_COLUMNS = ["serum_creatinine", "urea","dimethylamine", "phenylacetylglutamine",
    "trigonellin","lactate","citrate","hippurate","alanine","eGFR","time.TX"]
CATEGORICAL_COLUMNS = ["Sex", "Hypertension", "eGFR_bin","UA.Pro", "UA.Hb","Diabetes", "time.TX_bin"]



def tranform_data(features_df):
    
    features_df[NUMERICAL_COLUMNS] = features_df[NUMERICAL_COLUMNS].fillna(0)
    features_df[CATEGORICAL_COLUMNS] = features_df[CATEGORICAL_COLUMNS].fillna('unknown')
    features_df = pd.get_dummies(features, columns=CATEGORICAL_COLUMNS, drop_first=True)
    
    scaler = preprocessing.StandardScaler()
    scaled_features = scaler.fit_transform(features_df[NUMERICAL_COLUMNS])
    scaled_features = pd.DataFrame(scaled_features, columns=NUMERICAL_COLUMNS)
    
    features_df=features_df.drop(columns=NUMERICAL_COLUMNS)
    features_df = pd.concat([scaled_features,features_df], axis=1)
    
    return features_df
    
    

def load_dataset_transform(image_input_dir, batch_size, mode=tf.estimator.ModeKeys.EVAL):
    # Make a CSV dataset
    df = get_add_var(image_input_dir)
    df=df[CSV_COLUMNS]
    features,labels = df,df.pop(LABEL_COLUMN)
    features = transform_data(features)
    

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).cache()
        .shuffle(len(dataset))
        .batch(batch_size)

    # Shuffle and repeat for training
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=1000).repeat()

    # Take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


def build_model(hidden_units_1, hidden_units_2):
    model = Sequential()
    
    model.add(Dense(hidden_units_1, activation='relu', input_shape=(18,)))
    model.add(Dense(hidden_units_1, activation='relu'))
    model.add(Dense(hidden_units_2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_units_2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    
    return model

    
# Instantiate the HyperTune reporting object
hpt = hypertune.HyperTune()

# Reporting callback
class HPTCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        global hpt
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='auc',
            metric_value=logs['val_auc'],
            global_step=epoch)
        
        
def train_and_evaluate(args):
    model = build_model(args['nn_size1'], args['nn_size2'])

    trainds = load_dataset(args["train_data_path"], args["batch_size"], training=True)

    evalds = load_dataset(args["eval_data_path"], args["batch_size"], training=False)
    
    if args["eval_steps"]:
        evalds = evalds.take(count=args["eval_steps"])

    num_batches = args["batch_size"] * args["num_epochs"]
    steps_per_epoch = args["train_examples"] // args["batch_size"]
    checkpoint_path = os.path.join(args["output_dir"], "checkpoints/spectrain_csv_dnn")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True)

    history = model.fit(
        trainds,
        validation_data=evalds,
        epochs=args["batch_size"],
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        callbacks=[cp_callback, HPTCallback()])
    
    EXPORT_PATH = os.path.join(
        args["output_dir"], datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    tf.saved_model.save(
        obj=model, export_dir=EXPORT_PATH)  # with default serving function
    
    print("Exported trained model to {}".format(EXPORT_PATH))
