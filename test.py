from bert_serving.client import BertClient
bc = BertClient()
a=bc.encode([['我爱你',"你爱我","[PAD]"],['我爱你',"你爱我","[PAD]"]])
print(type(a))
print(a.shape)

'''
conda activate py36_tf1.12
bert-serving-start -model_dir=BERT-LSTM-CRF-BASE-word/bert_model/chinese_L-12_H-768_A-12 -pooling_layer=[-1] -pooling_strategy=FIRST_TOKEN
'''
