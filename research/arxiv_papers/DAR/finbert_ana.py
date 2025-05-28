import pandas as pd
import mindspore as ms
from mindnlp.transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
hf_token = ""

tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained(model_path, token=hf_token)

def predict_sentiment(text: str) -> float:
    inputs = tokenizer(text, return_tensors='ms')
    outputs = model(**inputs)
    logits = outputs.logits
    probs = ms.ops.Softmax()(logits)
    pos = probs[0][2].asnumpy().item()
    neg = probs[0][0].asnumpy().item()
    return pos - neg

df = pd.read_csv('input.csv', dtype={'Date': str, 'Text': str})

results = []
for idx, row in df.iterrows():
    date = row['Date']
    snippets = [s.strip() for s in row['Text'].split(',') if s.strip()]
    scores = [predict_sentiment(s) for s in snippets]
    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    results.append({
        'Date': date,
        'S_t': avg_score
    })

out_df = pd.DataFrame(results)
out_df.to_csv('daily_sentiment.csv', index=True)
