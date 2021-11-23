import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
from nltk.tokenize.treebank import TreebankWordDetokenizer
import spacy
from spacy.tokens import Doc
from copy import deepcopy
from utils import overlap, OSS
from tqdm import tqdm

class Compressor:
    def __init__(self, bert="bert-base-cased", device="cuda", spacy_name='en_core_web_sm'):
        self.tokenizer, self.model = BertTokenizer.from_pretrained(bert), BertForMaskedLM.from_pretrained(bert)
        self.device = torch.device(device)
        self.model = self.model.to(self.device); self.model.requires_grad = False
        self.nlp, self.nlp_dp = spacy.load(spacy_name), spacy.load(spacy_name)
        self.nlp_dp.tokenizer = lambda tokens: Doc(self.nlp_dp.vocab, tokens)
        self.detokenizer = TreebankWordDetokenizer()

    def NeighboringDistribution(self, tokens, neighbors):
        ids_batch, masks = [], []
        n_neighbor = len(neighbors)
        for idx in neighbors:
            tokens_masked = deepcopy(tokens)
            tokens_masked[idx] = self.tokenizer.mask_token
            items = self.tokenizer(' '.join(tokens_masked))
            ids_batch.append(items['input_ids']); masks.append(items['attention_mask']);

        max_len = max([len(ids) for ids in ids_batch])
        ids_batch_padded = [[self.tokenizer.pad_token_id for _ in range(max_len)] for ids in ids_batch]
        masks_padded = [[0 for _ in range(max_len)] for mask in masks]

        for idx in range(n_neighbor):
            ids_batch_padded[idx][:len(ids_batch[idx])] = ids_batch[idx]
            masks_padded[idx][:len(masks[idx])] = masks[idx]

        ids_batch_padded = torch.stack([torch.LongTensor(ids).to(self.device) for ids in ids_batch_padded], 0)
        masks_padded = torch.stack([torch.LongTensor(mask).to(self.device) for mask in masks_padded], 0)

        mask_pos = torch.argmax(ids_batch_padded.eq(self.tokenizer.mask_token_id).float(), 1)
        logits = torch.stack([logit[mask_pos[idx]] for idx, logit in enumerate(self.model(ids_batch_padded, masks_padded)['logits'].softmax(-1))], 0)
        return logits
    
    def NeighboringDistributionDivergence(self, tokens, span, start, end, decay_rate, biased, bias_rate):
        with torch.no_grad():
            tokens_ = tokens[:start] + span + tokens[end:]
            start_, end_ = start, start+len(span)
            neighbors = [idx for idx in range(len(tokens)) if idx < start or idx >= end]
            neighbors_ = [idx for idx in range(len(tokens_)) if idx < start_ or idx >= end_]
            sc, sc_ = self.NeighboringDistribution(tokens, neighbors), self.NeighboringDistribution(tokens_, neighbors_)
            w = np.array(list(np.arange(start)[::-1])+list(np.arange(len(tokens)-end)))
            w_ = w + (len(w) - w[::-1])
            w = torch.FloatTensor(np.power(decay_rate, w)) + torch.FloatTensor(np.power(decay_rate, w_))
            if biased:
                b = np.power(bias_rate, np.arange(len(w)))
                w = w * b
            w = w.to(self.device)
            ndd = ((sc_ * (sc_/sc).log()).sum(1)*w).sum(0)
            return ndd.item()

    def SpanSearch(self, sent, head, max_span, threshold, decay_rate, biased, bias_rate):
        spans = []
        length = len(sent)
        ids = list(np.arange(length))
        candidates = [(start, end) for start in range(length) for end in range(start+1, min(length, start+max_span))]
        bar = tqdm(candidates)
        bar.set_description("Compressing...")
        for candidate in bar:
            start, end = candidate
            ndd = self.NeighboringDistributionDivergence(sent, [], start, end, decay_rate, biased, bias_rate)
            if ndd < threshold:
                spans.append({'ids':np.arange(start, end), 'ndd':ndd})
        return spans
        
    def Compress(self, sent, max_itr=5, max_span=5, max_len=50, threshold=1.0, decay_rate=0.9, biased=False, bias_rate=0.98):
        logs = {}
        sent = [token.text for token in self.nlp(sent)]
        logs["Base"] = self.detokenizer.detokenize(sent)
        
        if len(sent) <= max_len:
            for itr in range(max_itr):
                head = [token.head.i for token in self.nlp_dp(sent)]
                spans = self.SpanSearch(sent, head, max_span, threshold, decay_rate, biased, bias_rate)
                spans = OSS(spans)
                if len(spans) == 0:break;
                span_ids = [idx for span in spans for idx in span['ids']]
                sent = [sent[idx] for idx in range(len(sent)) if idx not in span_ids]
                logs[f"Iter{itr}"] = self.detokenizer.detokenize(sent)
        return logs
