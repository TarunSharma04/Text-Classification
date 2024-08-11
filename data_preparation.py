import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(train_path, test_path):
    """
    Loads training and testing data from CSV files.

    Args:
    train_path (str): Path to the training CSV file.
    test_path (str): Path to the testing CSV file.

    Returns:
    pd.DataFrame: DataFrames containing training and testing data.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Combine 'Title' and 'Description' into a single column 'text'
    train_df['text'] = train_df['Title'] + " " + train_df['Description']
    test_df['text'] = test_df['Title'] + " " + test_df['Description']
    
    # Drop the original 'Title' and 'Description' columns
    train_df = train_df.drop(columns=['Title', 'Description'])
    test_df = test_df.drop(columns=['Title', 'Description'])
    
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Preprocesses text data using TF-IDF vectorization and adjusts labels to be zero-based.

    Args:
    train_df (pd.DataFrame): DataFrame containing training data.
    test_df (pd.DataFrame): DataFrame containing testing data.

    Returns:
    X_train_tfidf (sparse matrix): TF-IDF vectorized training data.
    X_test_tfidf (sparse matrix): TF-IDF vectorized testing data.
    y_train (pd.Series): Labels for training data.
    y_test (pd.Series): Labels for testing data.
    """
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # Fit and transform the training data, transform the testing data
    X_train_tfidf = vectorizer.fit_transform(train_df['text'])
    X_test_tfidf = vectorizer.transform(test_df['text'])
    
    # Adjust labels to be zero-based
    y_train = train_df['Class Index'] - 1
    y_test = test_df['Class Index'] - 1
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test

# Example usage
if __name__ == "__main__":
    train_df, test_df = load_data('train.csv', 'test.csv')
    X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess_data(train_df, test_df)
    print("Data loaded and preprocessed.")
