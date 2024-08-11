import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(input_shape, num_classes):
    """
    Builds a simple neural network model using TensorFlow.

    Args:
    input_shape (int): Number of features in the input data.
    num_classes (int): Number of output classes.

    Returns:
    model (tf.keras.Model): Compiled TensorFlow model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=5, batch_size=32, validation_split=0.1):
    """
    Trains the TensorFlow model.

    Args:
    model (tf.keras.Model): Compiled TensorFlow model.
    X_train (np.array): Training data features.
    y_train (np.array): Training data labels.
    epochs (int): Number of epochs to train.
    batch_size (int): Size of each training batch.
    validation_split (float): Fraction of training data to use for validation.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the TensorFlow model on test data.

    Args:
    model (tf.keras.Model): Trained TensorFlow model.
    X_test (np.array): Testing data features.
    y_test (np.array): Testing data labels.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

# Example usage
if __name__ == "__main__":
    import data_preparation
    
    # Load and preprocess data
    train_df, test_df = data_preparation.load_data('train.csv', 'test.csv')
    X_train_tfidf, X_test_tfidf, y_train, y_test = data_preparation.preprocess_data(train_df, test_df)
    
    # Build, train, and evaluate the model
    model = build_model(X_train_tfidf.shape[1], len(train_df['Class Index'].unique()))
    train_model(model, X_train_tfidf.toarray(), y_train)
    evaluate_model(model, X_test_tfidf.toarray(), y_test)
