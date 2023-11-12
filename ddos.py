import os

import keras.saving
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define model path
model_path = 'ddos_detection_model.keras'


@keras.saving.register_keras_serializable()
class DDOSDetectionModel(tf.keras.Model):
    """
    This TensorFlow-based model, implemented using the Keras API, is specifically designed for the detection of DDoS
    attacks. The architecture utilizes a sequence of densely connected neural layers,
    integrated with batch normalization and dropout techniques for performance and to mitigate overfitting.

    The model comprises three core dense layers with varying units (64, 32, and 16), each followed by batch
    normalization, a ReLU activation function, and dropout.

    Dropout rates are configurable upon initialization. The final output layer employs a sigmoid activation function,
    making the model suitable for binary classification of attack or not.

    Parameters:
        dropout_rate (float): The dropout rate applied to each dropout layer in the model. Default is 0.3.

    The model's `call` method defines the forward pass, applying each layer in sequence to the input data.

    Example usage:
        model = DDOSDetectionModel(dropout_rate=0.3)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32)

    Note:
        - Ensure the input data is preprocessed and scaled appropriately before feeding it into the model.
    """

    def __init__(self, dropout_rate=0.3):
        super(DDOSDetectionModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(None,))  # Explicit input layer
        self.dense1 = tf.keras.layers.Dense(64)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation('relu')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.dense2 = tf.keras.layers.Dense(32)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.activation2 = tf.keras.layers.Activation('relu')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.dense3 = tf.keras.layers.Dense(16)
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.activation3 = tf.keras.layers.Activation('relu')
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)

        x = self.dense3(x)
        x = self.batch_norm3(x)
        x = self.activation3(x)
        x = self.dropout3(x)

        return self.output_layer(x)


# Ensure TensorFlow has GPU available
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Warning: GPU not found, this may perform poorly.")


def load_model_if_exists(model_path):
    """
    Load the model if it exists, otherwise return None.
    :param model_path: Path to the model file.
    :return: The loaded model, or None if it does not exist.
    """
    if os.path.exists(model_path):
        print("Loading existing model.")
        custom_objects = {'DDOSDetectionModel': DDOSDetectionModel}
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        return None


def save_model(model, model_path):
    """
    Save the model to the specified path.
    :param model: Model instance
    :param model_path: Path model should be saved
    :return: None
    """
    model.save(model_path)
    print(f"Model saved to {model_path}.")

if __name__ == '__main__':
    # Load the dataset from CSV
    df = pd.read_csv('cicddos2019_dataset.csv', index_col=0)

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # Create a new column for the label Class (Attack or Benign)
    df = pd.get_dummies(df, columns=['Class'])

    # Splitting the dataset into features and labels
    # Drop the Label column from X, as it is not a feature
    X = df.drop(['Class_Attack', 'Class_Benign', 'Label'], axis=1)
    # Only the Label column is needed for the labels
    y = df['Class_Attack']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Attempt to load the model if it exists, otherwise create a new model and train it
    model = load_model_if_exists(model_path)
    if model is None:
        model = DDOSDetectionModel()

        # Compile the model
        # Use binary crossentropy as the loss function, as this is a binary classification problem ("attack or not")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=256, validation_split=0.2)

        # Save the trained model
        save_model(model, model_path)

        # Plot accuracy and loss over time (by default this is enabled)
        generate_plots = True
        if generate_plots:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig('model_accuracy.png')

            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig('model_loss.png')

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

    # Generate a confusion matrix (by default this is enabled)
    generate_confusion_matrix = True
    if generate_confusion_matrix:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # Predict the test data
        y_pred = model.predict(X_test_scaled)
        y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels

        # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)

        # Plotting the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')
