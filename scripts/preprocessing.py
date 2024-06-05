import os
import pandas as pd
import logging
import yaml
from google.colab import drive
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from lxml import etree as ET
import json
import csv
from concurrent.futures import ThreadPoolExecutor

# Load configuration
with open('/content/drive/MyDrive/your_project/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Mount Google Drive
drive.mount('/content/drive')

# Configure logging
logging.basicConfig(filename='/content/drive/MyDrive/your_project/logs/error.log', level=config['logging']['level'])
logger = logging.getLogger()

# Ensure NLTK data is available
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define stop words
stop_words = set(config['stop_words'])

# Initialize spacy 'en' model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.max_length = 70000000

def preprocess_text(text, chunk_size=1000000):
    """
    Clean and preprocess text for analysis.

    :param text: The text content to preprocess.
    :param chunk_size: The size of each chunk to split the text into.
    :return: Preprocessed text.
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    processed_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)
        processed_chunk = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        processed_chunks.extend(processed_chunk)
    return ' '.join(processed_chunks)

def extract_metadata_and_text(file_path, namespaces):
    """
    Extract metadata and text content from an XML file using XPath.

    :param file_path: The path to the XML file.
    :param namespaces: A dictionary of XML namespaces.
    :return: A dictionary containing metadata and the full text.
    """
    try:
        with open(file_path, 'r') as file:
            tree = ET.parse(file)
    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return None

    title_list = tree.xpath('//tei:sourceDesc//tei:biblFull//tei:titleStmt//tei:title/text()', namespaces=namespaces)
    title = title_list[0] if title_list else "Title not found"
    author_list = tree.xpath('//tei:sourceDesc//tei:biblFull//tei:titleStmt//tei:author/text()', namespaces=namespaces)
    author = author_list[0] if author_list else "Author not found"
    publication_date_list = tree.xpath('//tei:sourceDesc//tei:biblFull//tei:publicationStmt//tei:date/text()', namespaces=namespaces)
    publication_date = publication_date_list[0] if publication_date_list else "Publication date not found"
    place_of_publication_list = tree.xpath('//tei:sourceDesc//tei:biblFull//tei:publicationStmt//tei:pubPlace/text()', namespaces=namespaces)
    place_of_publication = place_of_publication_list[0] if place_of_publication_list else "Place of publication not found"
    subject_keywords = tree.xpath('//tei:textClass//tei:keywords//tei:term/text()', namespaces=namespaces)
    keywords_str = ', '.join(subject_keywords) if subject_keywords else "Keywords not found"
    full_text = ' '.join(tree.xpath('//tei:text//text()', namespaces=namespaces)).strip()

    metadata = {
        'Filename': os.path.basename(file_path),
        'Title': title,
        'Author': author,
        'Publication Date': publication_date,
        'Place of Publication': place_of_publication,
        'Subject Keywords': keywords_str,
        'FullText': full_text
    }

    return metadata

def load_checkpoints(checkpoint_path):
    try:
        with open(checkpoint_path, 'r', newline='') as f:
            reader = csv.reader(f)
            return dict(reader)
    except FileNotFoundError:
        return {}

def save_checkpoints(checkpoint_path, checkpoints):
    with open(checkpoint_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(checkpoints.items())

def validate_metadata(metadata):
    """
    Validate the extracted metadata.

    :param metadata: Metadata dictionary.
    :return: Boolean indicating whether the metadata is valid.
    """
    return bool(metadata)

def save_metadata(metadata_list, save_path, batch_number):
    """
    Save extracted metadata to a CSV file in a specified directory.

    :param metadata_list: A list of dictionaries containing metadata.
    :param save_path: The base path where the CSV directory is located.
    :param batch_number: The current batch number to create a unique filename.
    """
    output_dir = os.path.join(save_path, 'metadata_csv')
    os.makedirs(output_dir, exist_ok=True)
    csv_file_name = f'metadata_batch_{batch_number}.csv'
    full_csv_path = os.path.join(output_dir, csv_file_name)
    df = pd.DataFrame(metadata_list)
    df.to_csv(full_csv_path, index=False)
    print(f'Saved: {full_csv_path}')

def process_batches(base_path, namespaces, checkpoint_path):
    """
    Process XML files in batches, extract metadata, and save it to JSON files.

    :param base_path: The base directory where XML files are stored.
    :param namespaces: XML namespaces for XPath.
    :param checkpoint_path: Path to the checkpoint file.
    """
    checkpoints = load_checkpoints(checkpoint_path)
    all_files = sorted(glob.glob(os.path.join(base_path, '**/*.xml'), recursive=True))

    for file_path in all_files:
        if file_path in checkpoints:
            continue  # Skip already processed files

        metadata = extract_metadata_and_text(file_path, namespaces)
        if metadata and validate_metadata(metadata):
            json_path = file_path.replace('.xml', '.json').replace('.tei', '.json')
            with open(json_path, 'w') as f:
                json.dump(metadata, f)
            print(f'Processed and validated: {file_path}')
        else:
            print(f'Validation failed, skipped: {file_path}')

        checkpoints[file_path] = 'processed'
        save_checkpoints(checkpoint_path, checkpoints)

def main_processing_loop(base_path, namespaces, batch_size, progress_tracker_path, checkpoints_path):
    """
    Main loop to process XML files in batches.

    :param base_path: The base directory where XML files are stored.
    :param namespaces: XML namespaces for XPath.
    :param batch_size: Number of files to process in each batch.
    :param progress_tracker_path: Path to the progress tracker file.
    :param checkpoints_path: Path to the checkpoints file.
    """
    start_index = read_progress_tracker(progress_tracker_path)
    all_files = sorted(glob.glob(os.path.join(base_path, '**/*.xml'), recursive=True))
    total_files = len(all_files)
    processed_files = 0

    while start_index + processed_files < total_files:
        batch_files = all_files[start_index + processed_files:start_index + processed_files + batch_size]
        metadata_list = []
        for file_path in batch_files:
            metadata = extract_metadata_and_text(file_path, namespaces)
            if metadata:
                processed_text = preprocess_text(metadata['FullText'])
                metadata['ProcessedText'] = processed_text
                metadata_list.append(metadata)

        batch_number = (start_index + processed_files) // batch_size + 1
        save_metadata(metadata_list, config['data_path']['processed'], batch_number)
        processed_files += len(batch_files)
        with open(progress_tracker_path, 'w') as f:
            f.write(str(start_index + processed_files))

        print(f'Processed {start_index + processed_files}/{total_files} files.')

if __name__ == '__main__':
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    main_processing_loop(config['data_path']['raw'], namespaces, config['batch_size'], config['progress_tracker_path'], config['checkpoints_path'])
