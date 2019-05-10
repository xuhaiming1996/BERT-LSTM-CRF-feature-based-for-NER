from tokenization import FullTokenizer

tokenizer=FullTokenizer("./bert_model/chinese_L-12_H-768_A-12/vocab.txt")
print(tokenizer.tokenize("我 shi 许@#$%$%海明"))
print("".join(tokenizer.tokenize("我 shi 许海明")))