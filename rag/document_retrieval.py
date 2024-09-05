from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import numpy as np
from rank_bm25 import BM25Okapi
from csv2text.process_data import Csv2Text
import string
from copy import deepcopy
from underthesea import sent_tokenize

class DocumentRetrieval:
    def __init__(self):
        self.corpus = Csv2Text().meta_corpus
        self.embeddings_corpus = Csv2Text().embeddings_corpus
        self.tokenized_corpus = [self.split_text(doc["passage"]) for doc in (self.corpus)]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.embeddings = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
        
    def split_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.lower().split()
        words = [word for word in words if len(word.strip()) > 0]
        return words
    
    def retrieve(self, question, topk=50):
        """
        Get most relevant chunks to the question using combination of BM25 and semantic scores.
        """
        ## initialize query for each retriever (BM25 and semantic)
        tokenized_query = self.split_text(question)
        segmented_question = tokenize(question)
        question_emb = self.embeddings.encode([segmented_question])
        question_emb /= np.linalg.norm(question_emb, axis=1)[:, np.newaxis]

        ## get BM25 and semantic scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        semantic_scores = question_emb @ self.embeddings_corpus.T
        semantic_scores = semantic_scores[0]

        ## update chunks' scores.
        max_bm25_score = max(bm25_scores)
        min_bm25_score = min(bm25_scores)
        def normalize(x):
            return (x - min_bm25_score + 0.1) / \
            (max_bm25_score - min_bm25_score + 0.1)

        corpus_size = len(self.corpus)
        for i in range(corpus_size):
            self.corpus[i]["bm25_score"] = bm25_scores[i]
            self.corpus[i]["bm25_normed_score"] = normalize(bm25_scores[i])
            self.corpus[i]["semantic_score"] = semantic_scores[i]

        ## compute combined score (BM25 + semantic)
        for passage in self.corpus:
            passage["combined_score"] = passage["bm25_normed_score"] * 0.4 + \
                                        passage["semantic_score"] * 0.6

        ## sort passages by the combined score
        sorted_passages = sorted(self.corpus, key=lambda x: x["combined_score"], reverse=True)
        return sorted_passages[:topk]
    
    def extract_consecutive_subarray(self, numbers):
        subarrays = []
        current_subarray = []
        for num in numbers:
            if not current_subarray or num == current_subarray[-1] + 1:
                current_subarray.append(num)
            else:
                subarrays.append(current_subarray)
                current_subarray = [num]

        subarrays.append(current_subarray)  # Append the last subarray
        # print(subarrays)
        return subarrays
    
    def merge_contexts(self, passages):
        passages_sorted_by_id = sorted(passages, key=lambda x: x["id"], reverse=False)
        # psg_texts = [x["passage"].strip("Title: ").strip(x["title"]).strip()
        #              for x in passages_sorted_by_id]

        psg_ids = [x["id"] for x in passages_sorted_by_id]
        consecutive_ids = self.extract_consecutive_subarray(psg_ids)

        merged_contexts = []
        b = 0
        for ids in consecutive_ids:
            psgs = passages_sorted_by_id[b:b+len(ids)]
            # print(f'func2 - {psgs}')
            psg_texts = [x["passage"].strip("Title: ").strip(x["title"]).strip()
                        for x in psgs]
            merged = f"Title: {psgs[0]['title']}\n\n" + " ".join(psg_texts)
            b = b+len(ids)
            merged_contexts.append(dict(
                title=psgs[0]['title'],
                passage=merged,
                score=max([x["combined_score"] for x in psgs]),
                merged_from_ids=ids
            ))
        # print(f'merge text: {merged_contexts}')
        return merged_contexts
    
    def discard_contexts(self, passages):
        sorted_passages = sorted(passages, key=lambda x: x["score"], reverse=False)
        if len(sorted_passages) == 1:
            return sorted_passages
        else:
            shortened = deepcopy(sorted_passages)
            for i in range(len(sorted_passages) - 1):
                current, next = sorted_passages[i], sorted_passages[i+1]
                if next["score"] - current["score"] >= 0.05:
                    shortened = sorted_passages[i+1:]
            return shortened

    def expand_context(self, passage, n_sent=3):
        # psg_id = passage["id"]
        merged_from_ids = passage["merged_from_ids"]
        title = passage["title"]
        prev_id = merged_from_ids[0] - 1
        next_id = merged_from_ids[-1] + 1
        strip_title = lambda x: x["passage"].strip(f"Title: {x['title']}\n\n")

        texts = []
        if prev_id in range(0, len(self.corpus)):
            prev_psg = self.corpus[prev_id]
            if prev_psg["title"] == title:
                prev_text = strip_title(prev_psg)
                # prev_text = " ".join(prev_text.split()[-word_window:])
                prev_text = " ".join(sent_tokenize(prev_text)[-n_sent:])
                texts.append(prev_text)

        texts.append(strip_title(passage))

        if next_id in range(0, len(self.corpus)):
            next_psg = self.corpus[next_id]
            if next_psg["title"] == title:
                next_text = strip_title(next_psg)
                # next_text = " ".join(next_text.split()[:word_window])
                next_text = " ".join(sent_tokenize(next_text)[:n_sent])
                texts.append(next_text)

        expanded_text = " ".join(texts)
        expanded_text = f"Title: {title}\n{expanded_text}"
        new_passage = deepcopy(passage)
        new_passage["passage"] = expanded_text
        return new_passage

    def expand_contexts(self, passages):
        new_passages = [self.expand_context(passage) for passage in passages]
        return new_passages

    def collapse(self, passages):
        new_passages = deepcopy(passages)
        titles = {}
        for passage in new_passages:
            title = passage["title"]
            if not titles.get(title):
                titles[title] = [passage]
            else:
                titles[title].append(passage)
        best_passages = []
        for k, v in titles.items():
            best_passage = max(v, key= lambda x: x["score"])
            best_passages.append(best_passage)
        return best_passages
    def retrieve_documents(self, question, topk=3):
        passages = self.retrieve(question, topk)
        merged_passages = self.merge_contexts(passages)
        shortened_passages = self.discard_contexts(merged_passages)
        expanded_passages = self.expand_contexts(shortened_passages)
        final_passages = self.collapse(expanded_passages)
        text =""
        for doc in final_passages:
            text += doc['passage'].strip("Title: ").strip(doc["title"]).strip() + '\n'
        return text
    
# retriever = DocumentRetrieval()
# question = "Có bao nhiêu loại nồi cơm điện"
# print(retriever.retrieve_documents(question))