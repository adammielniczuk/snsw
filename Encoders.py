import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from transformers import GPT2Tokenizer, GPT2Model, T5Tokenizer, T5Model, LongformerTokenizer, LongformerModel
import torch

class BaseEncoder:
    def encode(self, text):
        raise NotImplementedError("Subclasses should implement this method")

#Baseline normal distirbution
class NormalEncoder(BaseEncoder):
    def __init__(self, feature_shape, mean=0, std=1):
        """
        Initialize the NormalEncoder with a specific feature shape and distribution parameters.
        
        Args:
            feature_shape (tuple): The shape of the feature to be encoded.
            mean (float): The mean of the normal distribution. Default is 0.
            std (float): The standard deviation of the normal distribution. Default is 1.
        """
        if not isinstance(feature_shape, tuple):
            raise ValueError("feature_shape must be a tuple.")
        self.feature_shape = feature_shape
        self.mean = mean
        self.std = std

    def encode(self, text):
        """
        Generate an array with values from a normal distribution with the shape 
        specified during initialization.
        
        Returns:
            np.ndarray: An array with values from the normal distribution.
        """
        return np.random.normal(loc=self.mean, scale=self.std, size=self.feature_shape)

#Baseline uniform 0 distirbution
class DummyEncoder:
    def __init__(self, feature_shape):
        """
        Initialize the DummyEncoder with a specific feature shape.
        
        Args:
            feature_shape (tuple): The shape of the feature to be encoded.
        """
        if not isinstance(feature_shape, tuple):
            raise ValueError("feature_shape must be a tuple.")
        self.feature_shape = feature_shape

    def encode(self, text):
        """
        Generate an array of zeros with the shape specified during initialization.
        
        Returns:
            np.ndarray: An array of zeros with the initialized shape.
        """
        return np.zeros(self.feature_shape)

# Possible models: (sentence-transformers/all-MiniLM-L6-v2, sentence-transformers/all-MiniLM-L12-v2, 
# sentence-transformers/all-mpnet-base-v2, sentence-transformers/paraphrase-xlm-r-multilingual-v1, 
# sentence-transformers/distilbert-base-nli-stsb-mean-tokens, sentence-transformers/bert-base-nli-mean-tokens,
# sentence-transformers/roberta-base-nli-mean-tokens, sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens,
# sentence-transformers/distiluse-base-multilingual-cased-v2)
class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, model_name):
        """
        Initialize the SentenceTransformer encoder.
        
        Args:
            model_name (str): The name of the model to load.
        """
        self.lm_encoder = SentenceTransformer(model_name)

    def encode(self, text):
        """
        Encode the input text using SentenceTransformer model.
        
        Returns:
            np.ndarray: Encoded feature vector.
        """
        return self.lm_encoder.encode(text)

# GPT2 Model
class GPT_2Encoder(BaseEncoder):
    def __init__(self):
        """
        Initialize the GPT_2Encoder encoder.
        
        Args:
            model_name (str): The name of the model to load.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2Model.from_pretrained("gpt2")

    def encode(self, text):
        """
        Encode the input text using GPT_2 model.
        
        Returns:
            np.ndarray: Encoded feature vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        output = last_hidden_states.mean(dim=1).detach().numpy()
        return output

# OpenAI encoder, text-embedding-3-small
class OpenAIEncoder(BaseEncoder):
    def __init__(self, model):
        """
        Initialize the OpenAI API encoder.
        """
        load_dotenv()
        self.client = OpenAI()
        self.model = model

    def encode(self, text):
        """
        Get embedding from OpenAI API.
        
        Args:
            text (str): The text to be encoded.
            model (str): The model to use for encoding.
        
        Returns:
            np.ndarray: The embedding for the input text.
        """
        text = text.replace("\n", " ")
        return np.array(self.client.embeddings.create(input = [text], model=self.model).data[0].embedding)

# T5 model
class T5Encoder(BaseEncoder):
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5Model.from_pretrained("t5-small")

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model.encoder(**inputs)
        output = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return output

# longformer-base-4096 model    
class LongformBase4096(BaseEncoder):
    def __init__(self):
        self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)
        outputs = self.model(**inputs)
        output = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return output
    
# Possible models: (microsoft/deberta-v3-large, google/electra-base-discriminator)
class HiddenStateTransformer(BaseEncoder):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        output = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return output

class TextEncoder:
    def __init__(self, encoder: BaseEncoder, target_dim: int = None):
        """
        Initialize the TextEncoder with a specific encoder and an optional target output size.
        
        Args:
            encoder (BaseEncoder): The encoder object that will be used for encoding.
            target_dim (int, optional): The desired output size of the encoding. Default is None.
        """
        self.encoder = encoder
        self.target_dim = target_dim

    def encode(self, text):
        """
        Use the encoder to encode the text, and adjust the size to the target dimension if specified.
        
        Args:
            text (str): The text to be encoded.
        
        Returns:
            np.ndarray: The encoded feature vector, possibly resized.
        """
        encoding = self.encoder.encode(text)
        if encoding.shape[0] == 1:
            encoding = encoding.reshape(-1)
        if self.target_dim is not None:
            if len(encoding) > self.target_dim:
                encoding = encoding[:self.target_dim]
            elif len(encoding) < self.target_dim:
                encoding = np.pad(encoding, (0, self.target_dim - len(encoding)))
        return encoding