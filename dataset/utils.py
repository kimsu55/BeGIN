from huggingface_hub import hf_hub_download
import shutil
import os.path as osp
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

def download_hf_file(repo_id,
                     filename,
                     local_dir,
                     subfolder=None,
                     repo_type="dataset",
                     cache_dir=None,
                     ):

    hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename, repo_type=repo_type,
                    local_dir=local_dir,  cache_dir=cache_dir, force_download=True)
    if subfolder is not None:
        shutil.move(osp.join(local_dir, subfolder, filename), osp.join(local_dir, filename))
        shutil.rmtree(osp.join(local_dir, subfolder))
    return osp.join(local_dir, filename)


def generate_node_features(text_list, num_bags=1703):
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    """
    Generate node features using a Bag-of-Words (BoW) representation.

    Parameters:
        text_list (list of str): A list of textual data (e.g., web page content).
        num_bags (int): The maximum number of words in the vocabulary (default: 1703).

    Returns:
        node_features (array): A binary Bag-of-Words matrix (0 or 1),
                               where each row represents a text and each column represents a word in the vocabulary.
    """
    # Define stopwords
    stop_words = set(stopwords.words('english'))

    # Preprocess each text: tokenize, remove stopwords, and keep only alphabetic tokens
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
        return ' '.join([word for word in tokens if word.isalpha() and word not in stop_words])
        # return ' '.join([word for word in tokens if word.isalpha()])

    text_cleaned = [preprocess_text(text) for text in text_list]

    # Create a CountVectorizer for binary Bag-of-Words representation
    vectorizer = CountVectorizer(max_features=num_bags, binary=True)

    # Fit and transform the cleaned text data
    node_features = vectorizer.fit_transform(text_cleaned).toarray()

    return node_features



