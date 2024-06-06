import pandas as pd
import re
import os
import json
import yaml
from google.colab import drive
import nltk
from nltk.corpus import stopwords

# Load configuration
with open('/content/drive/MyDrive/EEBO/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Mount Google Drive
drive.mount('/content/drive')

# Download NLTK stopwords
nltk.download('stopwords')

# Define stop words, including those from the config file
modern_stopwords = set(stopwords.words('english'))
early_modern_english_stopwords = set(config['stop_words'])
all_stopwords = modern_stopwords.union(early_modern_english_stopwords)

def clean_text(text):
    """
    Clean and normalize text.
    
    :param text: The text to clean.
    :return: Cleaned text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

def preprocess_existing_text(text):
    """
    Clean and preprocess existing processed text for further analysis.
    
    :param text: The text content to preprocess.
    :return: Cleaned and preprocessed text.
    """
    text = clean_text(text)  # Perform initial cleaning and normalization
    tokens = text.split()
    tokens = [word for word in tokens if word not in all_stopwords]
    return ' '.join(tokens)

def save_progress(batch_number, progress_tracker_path):
    """
    Save the progress of the current batch number.
    
    :param batch_number: The current batch number.
    :param progress_tracker_path: Path to the progress tracker file.
    """
    with open(progress_tracker_path, 'w') as f:
        json.dump(batch_number, f)

def load_progress(progress_tracker_path):
    """
    Load the progress from the progress tracker file.
    
    :param progress_tracker_path: Path to the progress tracker file.
    :return: The last processed batch number.
    """
    try:
        with open(progress_tracker_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return 0

def process_existing_csv_files(input_directory, output_directory, batch_size, progress_tracker_path):
    """
    Process existing CSV files by applying additional cleaning on the 'ProcessedText' column in batches.
    
    :param input_directory: Path to the directory containing preprocessed CSV files.
    :param output_directory: Path to the directory where cleaned files will be saved.
    :param batch_size: Number of files to process in each batch.
    :param progress_tracker_path: Path to the progress tracker file.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    all_files = []
    for root, dirs, files in os.walk(input_directory):
        for file_name in files:
            if file_name.endswith('.csv'):
                all_files.append(os.path.join(root, file_name))

    last_processed_batch = load_progress(progress_tracker_path)
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")

    for batch_start in range(last_processed_batch, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = all_files[batch_start:batch_end]
        print(f"Processing batch {batch_start // batch_size + 1} with files {batch_start} to {batch_end}")

        for file_path in batch_files:
            df = pd.read_csv(file_path)

            # Apply cleaning to the 'ProcessedText' column
            df['ProcessedText'] = df['ProcessedText'].apply(preprocess_existing_text)

            # Save the cleaned DataFrame to a new CSV file, preserving directory structure
            relative_path = os.path.relpath(file_path, input_directory)
            output_file_path = os.path.join(output_directory, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            df.to_csv(output_file_path, index=False)
            print(f"Processed and saved: {output_file_path}")

        save_progress(batch_start + batch_size, progress_tracker_path)

# Load paths and parameters from config
input_directory = config['data_path']['raw']
output_directory = config['data_path']['processed']
progress_tracker_path = config['progress_tracker_path']
batch_size = config['batch_size']

# Process existing CSV files in batches
process_existing_csv_files(input_directory, output_directory, batch_size, progress_tracker_path)
