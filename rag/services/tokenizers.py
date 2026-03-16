import jieba

def jieba_tokenizer(text):
    return list(jieba.cut(text))