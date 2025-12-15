# rag_evaluation_metrics.py
"""
Evaluation metrics for RAG experimental framework

Measures three core dimensions:
1. Information Flow Integrity (retrieval quality)
2. Retrieval-Generation Coupling (how well retrieval translates to answers)
3. Hallucination vs Coverage Trade-off (groundedness vs completeness)
"""

import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# ============================================
# 1. INFORMATION FLOW INTEGRITY
# ============================================

def hit_rate(y_true, y_pred):
    """
    Measures if ANY relevant document was retrieved
    """
    hits = 0
    for true_docs, pred_docs in zip(y_true, y_pred):
        if any(doc in true_docs for doc in pred_docs):
            hits += 1
    return hits / len(y_true) if len(y_true) > 0 else 0

def mean_reciprocal_rank(y_true, y_pred):
    """
    Measures ranking quality - how early relevant docs appear
    """
    reciprocal_ranks = []
    for true_docs, pred_docs in zip(y_true, y_pred):
        rr = 0
        for rank, doc in enumerate(pred_docs, start=1):
            if doc in true_docs:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)
    return np.mean(reciprocal_ranks)

def precision_recall_at_k(y_true, y_pred, k=5):
    """
    Core retrieval metrics
    """
    recall_list = []
    precision_list = []
    
    for true_docs, pred_docs in zip(y_true, y_pred):
        top_k = pred_docs[:k]
        hits = len([doc for doc in top_k if doc in true_docs])
        
        recall_list.append(hits / len(true_docs) if len(true_docs) > 0 else 0)
        precision_list.append(hits / k if k > 0 else 0)
    
    return {
        f"Recall@{k}": np.mean(recall_list),
        f"Precision@{k}": np.mean(precision_list)
    }

def ndcg_at_k(y_true, y_pred, k=5):
    """
    Normalized Discounted Cumulative Gain - ranking quality
    """
    ndcg_scores = []
    
    for true_docs, pred_docs in zip(y_true, y_pred):
        pred_docs_k = pred_docs[:k]
        
        # DCG
        dcg = sum([
            1 / np.log2(idx + 2) if doc in true_docs else 0 
            for idx, doc in enumerate(pred_docs_k)
        ])
        
        # IDCG
        ideal_docs_k = true_docs[:k]
        idcg = sum([1 / np.log2(idx + 2) for idx, _ in enumerate(ideal_docs_k)])
        
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
    
    return np.mean(ndcg_scores)

# ============================================
# 2. RETRIEVAL-GENERATION COUPLING
# ============================================

def context_utilization(responses, retrieved_docs, top_k=5):
    """
    CRITICAL METRIC: How much of the retrieved context is used in the answer?
    Measures retrieval → generation coupling
    """
    scores = []
    
    for resp, docs in zip(responses, retrieved_docs):
        resp_words = set(nltk.word_tokenize(resp.lower()))
        doc_words = set(
            word for d in docs[:top_k] 
            for word in nltk.word_tokenize(d.lower())
        )
        
        overlap = len(resp_words & doc_words) / len(resp_words) if len(resp_words) > 0 else 0
        scores.append(overlap)
    
    return np.mean(scores)

def answer_relevance(responses, queries):
    """
    Does the answer actually address the query?
    """
    scores = []
    
    for resp, query in zip(responses, queries):
        resp_words = set(nltk.word_tokenize(resp.lower()))
        query_words = set(nltk.word_tokenize(query.lower()))
        
        overlap = len(resp_words & query_words)
        scores.append(overlap / len(query_words) if len(query_words) > 0 else 0)
    
    return np.mean(scores)

def retrieval_generation_correlation(retrieval_scores, generation_scores):
    """
    META-METRIC: Do better retrievals lead to better answers?
    This is the KEY insight of your project
    """
    if len(retrieval_scores) < 2 or len(generation_scores) < 2:
        return 0.0
    
    correlation = np.corrcoef(retrieval_scores, generation_scores)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

# ============================================
# 3. HALLUCINATION VS COVERAGE TRADE-OFF
# ============================================

def groundedness(responses, retrieved_docs, top_k=5):
    """
    CRITICAL: What % of the answer is supported by retrieved docs?
    High groundedness = low hallucination risk
    """
    scores = []
    
    for resp, docs in zip(responses, retrieved_docs):
        resp_words = set(nltk.word_tokenize(resp.lower()))
        doc_words = set(
            word for d in docs[:top_k] 
            for word in nltk.word_tokenize(d.lower())
        )
        
        overlap = len(resp_words & doc_words) / len(resp_words) if len(resp_words) > 0 else 0
        scores.append(overlap)
    
    return np.mean(scores)

def hallucination_rate(responses, retrieved_docs, top_k=5):
    """
    CRITICAL: What % of the answer is NOT in the retrieved docs?
    Direct measure of unsupported claims
    """
    rates = []
    
    for resp, docs in zip(responses, retrieved_docs):
        resp_words = set(nltk.word_tokenize(resp.lower()))
        doc_words = set(
            word for d in docs[:top_k] 
            for word in nltk.word_tokenize(d.lower())
        )
        
        unsupported = len(resp_words - doc_words)
        rates.append(unsupported / len(resp_words) if len(resp_words) > 0 else 0)
    
    return np.mean(rates)

def coverage_score(responses, retrieved_docs, top_k=5):
    """
    What % of retrieved information made it into the answer?
    Measures information completeness
    """
    scores = []
    
    for resp, docs in zip(responses, retrieved_docs):
        resp_words = set(nltk.word_tokenize(resp.lower()))
        doc_words = set(
            word for d in docs[:top_k] 
            for word in nltk.word_tokenize(d.lower())
        )
        
        coverage = len(resp_words & doc_words) / len(doc_words) if len(doc_words) > 0 else 0
        scores.append(coverage)
    
    return np.mean(scores)

# ============================================
# 4. GENERATION QUALITY METRICS
# ============================================

def fluency_score(responses):
    """
    Readability and sentence structure quality
    """
    readability_scores = []
    coherence_scores = []
    
    for resp in responses:
        try:
            readability_scores.append(flesch_reading_ease(resp))
        except:
            readability_scores.append(0)
        
        sentences = nltk.sent_tokenize(resp)
        words = nltk.word_tokenize(resp)
        coherence = len(words) / len(sentences) if len(sentences) > 0 else 0
        coherence_scores.append(coherence)
    
    return {
        "Readability (Flesch)": np.mean(readability_scores),
        "Coherence (words/sentence)": np.mean(coherence_scores)
    }

def answer_length_stats(responses):
    """
    Simple length metrics
    """
    lengths = [len(resp.split()) for resp in responses]
    return {
        "Avg Answer Length": np.mean(lengths),
        "Min Length": np.min(lengths),
        "Max Length": np.max(lengths)
    }

# ============================================
# 5. COMPREHENSIVE EVALUATION WRAPPER
# ============================================

def evaluate_pipeline(queries, responses, retrieved_docs, top_k=5):
    """
    Complete evaluation suite measuring all three dimensions
    
    Returns comprehensive metrics for scientific analysis
    """
    metrics = {}
    
    # 1. Information Flow Integrity (Retrieval Quality)
    metrics["retrieval"] = {
        "Hit Rate": hit_rate(retrieved_docs, retrieved_docs),
        "MRR": mean_reciprocal_rank(retrieved_docs, retrieved_docs),
        "nDCG@5": ndcg_at_k(retrieved_docs, retrieved_docs, k=top_k),
    }
    metrics["retrieval"].update(precision_recall_at_k(retrieved_docs, retrieved_docs, k=top_k))
    
    # 2. Retrieval-Generation Coupling
    metrics["coupling"] = {
        "Context Utilization": context_utilization(responses, retrieved_docs, top_k),
        "Answer Relevance": answer_relevance(responses, queries),
    }
    
    # 3. Hallucination vs Coverage Trade-off
    metrics["faithfulness"] = {
        "Groundedness": groundedness(responses, retrieved_docs, top_k),
        "Hallucination Rate": hallucination_rate(responses, retrieved_docs, top_k),
        "Coverage Score": coverage_score(responses, retrieved_docs, top_k),
    }
    
    # 4. Generation Quality
    metrics["generation"] = fluency_score(responses)
    metrics["generation"].update(answer_length_stats(responses))
    
    # 5. Overall Score (weighted composite)
    metrics["overall_score"] = (
        metrics["coupling"]["Context Utilization"] * 0.3 +
        metrics["faithfulness"]["Groundedness"] * 0.3 +
        (1 - metrics["faithfulness"]["Hallucination Rate"]) * 0.2 +
        metrics["coupling"]["Answer Relevance"] * 0.2
    )
    
    return metrics

def compare_pipelines(pipeline_results):
    """
    Compare multiple pipeline results and identify best performer
    
    Returns ranking and detailed comparison
    """
    comparison = {}
    
    for pipeline_name, metrics in pipeline_results.items():
        comparison[pipeline_name] = {
            "Overall Score": metrics["overall_score"],
            "Groundedness": metrics["faithfulness"]["Groundedness"],
            "Context Use": metrics["coupling"]["Context Utilization"],
            "Hallucination": metrics["faithfulness"]["Hallucination Rate"],
        }
    
    # Rank by overall score
    ranked = sorted(
        comparison.items(), 
        key=lambda x: x[1]["Overall Score"], 
        reverse=True
    )
    
    return ranked, comparison