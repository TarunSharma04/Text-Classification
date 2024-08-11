def run_tensorflow_model():
    """
    Runs the TensorFlow model pipeline: loads data, preprocesses data, builds the model, trains the model, and evaluates the model.
    """
    import data_preparation
    import tensorflow_model
    
    # Load and preprocess data
    train_df, test_df = data_preparation.load_data('train.csv', 'test.csv')
    X_train_tfidf, X_test_tfidf, y_train, y_test = data_preparation.preprocess_data(train_df, test_df)
    
    # Build, train, and evaluate the model
    model = tensorflow_model.build_model(X_train_tfidf.shape[1], len(train_df['Class Index'].unique()))
    tensorflow_model.train_model(model, X_train_tfidf.toarray(), y_train)
    tensorflow_model.evaluate_model(model, X_test_tfidf.toarray(), y_test)

def run_transformers_model():
    """
    Runs the Hugging Face Transformers model pipeline: loads data, preprocesses data, builds the model, trains the model, and evaluates the model.
    """
    import data_preparation
    import transformers_model
    import tensorflow as tf
    
    # Load and preprocess data
    train_df, test_df = data_preparation.load_data('train.csv', 'test.csv')
    _, _, y_train, y_test = data_preparation.preprocess_data(train_df, test_df)
    
    # Build BERT model
    tokenizer, model = transformers_model.build_transformers_model(len(train_df['Class Index'].unique()))
    
    # Tokenize data
    train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=512)
    
    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(1000).batch(16)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(16)
    
    # Train and evaluate BERT model
    transformers_model.train_transformers_model(model, train_dataset)
    transformers_model.evaluate_transformers_model(model, test_dataset)

