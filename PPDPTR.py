from typing import Dict, List
from textblob import TextBlob
import textstat
import yake
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from tqdm.auto import tqdm
import string

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

STOP = set(nltk.corpus.stopwords.words("english"))
PUNCT = set(string.punctuation)

def sentence_formality(text: str) -> float:
    if not text: return 0.0
    pos_tags = TextBlob(text).tags
    formal_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'IN', 'DT'}
    informal_tags = {'PRP', 'PRP$', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'UH'}
    total_words = len(pos_tags)
    if total_words == 0: return 0.0
    formal_count = sum(1 for _, tag in pos_tags if tag in formal_tags)
    informal_count = sum(1 for _, tag in pos_tags if tag in informal_tags)
    # Heylighen and Dewaele’s Formality Formula (1999)
    f_score = ((formal_count - informal_count) / total_words) * 50 + 50
    return round(f_score, 2)

def formality_level(text: str) -> str:
    if not text: return "NEUTRAL"
    score = sentence_formality(text)
    if score > 75: return "VERY FORMAL"
    elif score > 55: return "FORMAL"
    elif score >= 45: return "NEUTRAL"
    elif score >= 25: return "INFORMAL"
    else: return "VERY INFORMAL"

def keyword_extraction(text: str, extractor) -> list[str]:
    if not text: return ["N/A"]
    try:
        keywords = extractor.extract_keywords(text)
        return [kw for kw, _ in keywords if kw not in STOP and kw not in PUNCT] if keywords else ["N/A"]
    except Exception:
        return ["N/A"]
    
def number_of_words(text: str) -> int:
    return len(TextBlob(text).words) if text else 0

def number_of_sentences(text: str) -> int:
    return len(TextBlob(text).sentences) if text else 0

def sentiment_category(text: str) -> str:
    if not text: return "N/A"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.6: return "VERY POSITIVE"
    elif polarity > 0.2: return "POSITIVE"
    elif polarity >= -0.2: return "NEUTRAL"
    elif polarity >= -0.6: return "NEGATIVE"
    else: return "VERY NEGATIVE"

def subjectivity_category(text: str) -> str:
    if not text: return "N/A"
    subjectivity = TextBlob(text).sentiment.subjectivity
    if subjectivity > 0.75: return "VERY SUBJECTIVE"
    elif subjectivity > 0.5: return "SUBJECTIVE"
    elif subjectivity > 0.25: return "OBJECTIVE"
    else: return "VERY OBJECTIVE"

def readability_level(text: str) -> str:
    if not text: return "N/A"
    score = textstat.flesch_kincaid_grade(text)
    if score >= 16: return "VERY DIFFICULT (Postgraduate): Use long, complex sentence structures and advanced, polysyllabic words."
    elif score >= 13: return "DIFFICULT (College): Use complex sentences and a broader vocabulary."
    elif score >= 10: return "MEDIUM DIFFICULT (10th-12th Grade): Use straightforward sentences of varied length."
    elif score >= 8: return "STANDARD (8th-9th Grade): Use clear sentences of average length."
    elif score >= 6: return "EASY (6th-7th Grade): Use short sentences and common words."
    else: return "VERY EASY (5th Grade & Below): Use extremely short sentences and very common, single-syllable words."

def latent_dirichlet_allocation(text: str, num_topics=3) -> list:
    # Identifies topics using LDA and returns only the words for the first topic (topic 0).
    if not text:
        return [] 

    try:
        sentences = sent_tokenize(text)
        tokenized_docs = [word_tokenize(s.lower()) for s in sentences]
        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        corpus = [c for c in corpus if c]
        if not corpus:
            return []
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=5, random_state=42)
        topic_0 = lda_model.show_topic(0)
        return [word for word, prob in topic_0 if word not in STOP and word not in PUNCT]
    except Exception:
        return []
    

class PP:
    def __init__(self, model_checkpoint="meta-llama/Llama-3.2-3B-Instruct", hf_token=None):
        # HuggingFace token required if using gated models!
        self.yake_extractor = yake.KeywordExtractor(n=1, top=5, dedupLim=0.9)

        if torch.cuda.is_available() == True:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.gen_model = AutoModelForCausalLM.from_pretrained(model_checkpoint, token=hf_token, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, token=hf_token, padding_side='left')
        
        self.pipe = pipeline(
            "text-generation",
            model=self.gen_model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id[0]

    def get_text_features(self, text: str, use_advanced_features: bool) -> dict:
        features = {
            'sentiment': sentiment_category(text),
            'subjectivity': subjectivity_category(text),
            'formality': formality_level(text),
            'word_count': number_of_words(text),
            'sentence_count': number_of_sentences(text),
            'readability': readability_level(text),
        }
        if use_advanced_features:
            features['keywords/topics'] = list(set(keyword_extraction(text, self.yake_extractor)+latent_dirichlet_allocation(text)))
        return features
    
    def create_postprocessing_prompt(self, dp_text: str, features: dict, use_advanced_features: bool) -> str:
        if use_advanced_features:
            x = "\n- **Topics:** {}\n".format(features.get('keywords/topics'))
        else:
            x = ""
        pass

        prompt_content = f"""
            You are a meticulous text editor. Your job is to rewrite the 'Noisy Text' to be fluent, clear, and grammatically correct.
            You must strictly and equally follow ALL of the 'Correction Rules' provided. No single rule is more important than another.
            Do NOT hallucinate any new information, keep similar word and sentence count, as well as readability level.
            Output ONLY the refined text without any explanation or preamble.

            ---
            **Comprehensive Example**
            ---
            This example shows a text that is too long, too formal, and has the wrong sentiment. The refined text corrects ALL of these errors to match the rules.

            ### Correction Rules
            - **Topics:** ['super', 'place', 'parking', 'great', 'meet']
            - **Sentiment:** POSITIVE
            - **Subjectivity:** SUBJECTIVE
            - **Formality:** INFORMAL
            - **Readability:** EASY (6th-7th Grade)
            - **Word Count:** 25 words
            - **Sentence Count:** 2 sentences

            ### Noisy Text
            Notwithstanding the foregoing, the establishment provided a suboptimal user experience. The vehicular parking situation was profoundly inconvenient, and the ambient illumination was insufficient for proper identification of my associate, precipitating our premature departure.

            ### Refined Text (Adhering to all rules)
            This was a great place to meet up! The parking was super easy and it was so well-lit that we found each other right away.

            ---
            **Your Task**
            ---

            ### Correction Rules{x}
            - **Sentiment:** {features.get('sentiment')}
            - **Subjectivity:** {features.get('subjectivity')}
            - **Formality:** {features.get('formality')}
            - **Readability:** {features.get('readability')}
            - **Word Count:** {features.get('word_count')} words
            - **Sentence Count:** {features.get('sentence_count')} sentences

            ### Noisy Text
            {dp_text}

            ### Refined Text (Adhering to all rules)
            """
        
        return prompt_content.strip()

    def postprocess(self, original_text: str, dp_text: str, use_advanced_features=False) -> str:
        features = self.get_text_features(text=original_text, use_advanced_features=use_advanced_features)
        prompt = self.create_postprocessing_prompt(dp_text=dp_text, features=features, use_advanced_features=use_advanced_features)

        outputs = self.pipe(
                prompt,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=int(len(self.tokenizer.encode(dp_text, return_tensors="pt")[0]))
            )
        generated = outputs[0]["generated_text"]
        generated = generated.split("### Refined Text (Adhering to all rules)")[-1].strip().replace("\n", "").split("###")[-1].strip()
        return generated

    def postprocess_batch(self, original_texts: str, dp_texts: str, use_advanced_features=False, batch_size=2) -> list:
        features = [self.get_text_features(text=x, use_advanced_features=use_advanced_features) for x in original_texts]
        prompts = [self.create_postprocessing_prompt(dp_text=dp_texts[i], features=x, use_advanced_features=use_advanced_features) for i, x in enumerate(features)]

        batch = ListDataset(prompts)
        generated = []
        for res in tqdm(self.pipe(batch, max_new_tokens=256, batch_size=batch_size)):
            g = res[0]["generated_text"]
            g = g.split("### Refined Text (Adhering to all rules)")[-1].strip().replace("\n", "").split("###")[-1].strip()
            generated.append(g)
        return generated