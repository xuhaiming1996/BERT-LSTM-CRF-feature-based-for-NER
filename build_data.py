from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM,PAD, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test.py) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word(lowercase=False)

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    # test.py  = CoNLLDataset(config.filename_test, processing_word)  后面需要吧测试集的 也加进来
    train = CoNLLDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev])
    # 这里先不加 get_glove_vocab
    # vocab_glove = get_glove_vocab(config.filename_glove)

    # vocab = vocab_words & vocab_glove
    vocab = vocab_words

    vocab.add(UNK)
    vocab.add(PAD)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    # vocab = load_vocab(config.filename_words)
    # export_trimmed_glove_vectors(vocab, config.filename_glove,config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars_train = get_char_vocab(train)

    dev = CoNLLDataset(config.filename_dev)
    vocab_chars_dev = get_char_vocab(dev)
    vocab_chars_train_dev=list(vocab_chars_dev & vocab_chars_train)
    vocab_chars=[UNK,PAD,NUM]
    vocab_chars.extend(vocab_chars_train_dev)

    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
