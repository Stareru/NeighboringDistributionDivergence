

def overlap(ids1, ids2):
    return len([idx for idx in ids1 if idx in ids2])

def OSS(spans, metric_name='ndd'):
    return [span for span in spans if not (True in [overlap(span['ids'], span_['ids']) and span[metric_name] > span_[metric_name] for span_ in spans])]
