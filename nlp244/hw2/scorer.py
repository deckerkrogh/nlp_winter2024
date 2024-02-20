import pandas as pd
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.scheme import IOB2
from sklearn.metrics import f1_score 
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('scorer', description="tool to score IOB slot tagging/core relations")
    parser.add_argument('--gold', required=True, type=str, help='path to golds.csv')
    parser.add_argument('--pred', required=True, type=str, help='path to preds.csv')

    args = parser.parse_args()

    gold_df = pd.read_csv(args.gold).fillna('')
    pred_df = pd.read_csv(args.pred).fillna('')

    pred_rels = [[r for r in rels.split(' ')] if isinstance(rels, str) else '' for rels in pred_df['Core Relations'].tolist()]
    gold_rels = [[r for r in rels.split(' ')] if isinstance(rels, str) else '' for rels in gold_df['Core Relations'].tolist()]

    mlb = MultiLabelBinarizer()
    mlb.fit(gold_rels)
    pred_rels = mlb.transform(pred_rels)
    gold_rels = mlb.transform(gold_rels)
    print('Relations score: ', f1_score(gold_rels, pred_rels, average='micro'))

    if 'IOB Slot tags' in pred_df.columns:
        pred_tags = [[t.replace('_', '-') for t in tags.split(' ')] for tags in pred_df['IOB Slot tags'].tolist()]
        gold_tags = [[t.replace('_', '-') for t in tags.split(' ')] for tags in gold_df['IOB Slot tags'].tolist()]
        
        print('Tagging score: ', seq_f1_score(gold_tags, pred_tags, scheme=IOB2))
