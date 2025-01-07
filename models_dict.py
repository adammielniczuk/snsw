from Encoders import *
from transformers import DebertaV2Tokenizer, DebertaV2Model
from transformers import ElectraTokenizer, ElectraModel

Mapped_Models = {
    "normal_384" : (TextEncoder(NormalEncoder(feature_shape=(384,), mean=0.0, std=1.0), target_dim=None), 384),
    "zeros_384" : (TextEncoder(DummyEncoder(feature_shape=(384,)), target_dim=None), 384),
    "normal_512" : (TextEncoder(NormalEncoder(feature_shape=(512,), mean=0.0, std=1.0), target_dim=None), 512),
    "zeros_512" : (TextEncoder(DummyEncoder(feature_shape=(512,)), target_dim=None), 512),
    "normal_768" : (TextEncoder(NormalEncoder(feature_shape=(768,), mean=0.0, std=1.0), target_dim=None), 768),
    "zeros_768" : (TextEncoder(DummyEncoder(feature_shape=(768,)), target_dim=None), 768),
    "normal_1024" : (TextEncoder(NormalEncoder(feature_shape=(1024,), mean=0.0, std=1.0), target_dim=None), 1024),
    "zeros_1024" : (TextEncoder(DummyEncoder(feature_shape=(1024,)), target_dim=None), 1024),
    "normal_1536" : (TextEncoder(NormalEncoder(feature_shape=(1536,), mean=0.0, std=1.0), target_dim=None), 1536),
    "zeros_1536" : (TextEncoder(DummyEncoder(feature_shape=(1536,)), target_dim=None), 1536),

    "all_MiniLM_L6_v2" : (TextEncoder(SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2"), target_dim=None), 384),
    "all_MiniLM_L12_v2" : (TextEncoder(SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L12-v2"), target_dim=None), 384),
    "all_mpnet_base_v2" : (TextEncoder(SentenceTransformerEncoder("sentence-transformers/all-mpnet-base-v2"), target_dim=None), 768),
    "paraphrase_xlm_r_multilingual_v1" : (TextEncoder(SentenceTransformerEncoder("sentence-transformers/paraphrase-xlm-r-multilingual-v1"), target_dim=None), 768),
    "distilbert_base_nli_stsb_mean_tokens" : (TextEncoder(SentenceTransformerEncoder("sentence-transformers/distilbert-base-nli-stsb-mean-tokens"), target_dim=None), 768),
    "bert_base_nli_mean_tokens" : (TextEncoder(SentenceTransformerEncoder("sentence-transformers/bert-base-nli-mean-tokens"), target_dim=None), 768),
    "roberta_base_nli_mean_tokens" : (TextEncoder(SentenceTransformerEncoder("sentence-transformers/roberta-base-nli-mean-tokens"), target_dim=None), 768),
    "xlm_r_bert_base_nli_stsb_mean_tokens" : (TextEncoder(SentenceTransformerEncoder("sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens"), target_dim=None), 768),
    "distiluse_base_multilingual_cased_v2" : (TextEncoder(SentenceTransformerEncoder("sentence-transformers/distiluse-base-multilingual-cased-v2"), target_dim=None), 512),
    "t5_small" : (TextEncoder(T5Encoder(), target_dim=None), 512),
    "longformer_base_4096" : (TextEncoder(LongformBase4096(), target_dim=None), 768),
    "deberta_v3_large" : (TextEncoder(HiddenStateTransformer(tokenizer=DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large"), model=DebertaV2Model.from_pretrained("microsoft/deberta-v3-large")), target_dim=None), 1024),
    "electra" : (TextEncoder(HiddenStateTransformer(tokenizer=ElectraTokenizer.from_pretrained("google/electra-base-discriminator"), model=ElectraModel.from_pretrained("google/electra-base-discriminator")), target_dim=None), 768),
    "gpt2" : (TextEncoder(GPT_2Encoder()), 768),
    "openai" : (TextEncoder(OpenAIEncoder(model="text-embedding-3-small")), 1536),
}