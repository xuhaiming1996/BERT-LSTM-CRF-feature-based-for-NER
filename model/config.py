import os


from .general_utils import get_logger
from .data_utils import load_vocab, \
        get_processing_word,get_processing_pos,get_processing_ps


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # min. vocabulary
        self.vocab_words = load_vocab(self.filename_words)  #加载单词
        self.vocab_poses = load_vocab(self.filename_poses)  #加载 词性字典
        self.vocab_chars = load_vocab(self.filename_chars)
        self.vocab_ps = load_vocab(self.filename_ps)
        self.vocab_tags  = load_vocab(self.filename_tags)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.nposes     = len(self.vocab_poses)
        self.nps        = len(self.vocab_ps)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=False, chars=self.use_chars)

        self.processing_pos = get_processing_pos(self.vocab_poses,allow_PAD=True)
        self.processing_ps  = get_processing_ps(self.vocab_ps)
        print(self.vocab_tags)

        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        # self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
        #         if self.use_pretrained else None)


    # general config
    dir_output = "results/test.py/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_char = 100

    # glove files

    filename_train = "./data/train.train"  # test.py
    filename_dev ="./data/dev.train"
    filename_test = ""


    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "./data/words.txt"    #单词
    filename_poses = "./data/postag.txt"   #词性
    filename_ps = "./data/ps.txt"          # 字符
    filename_chars = "./data/chars.txt"    #字符
    filename_tags = "./data/tags.txt"      #标签




    # training
    train_embeddings = True
    nepochs          = 30
    dropout          = 0.7
    batch_size       = 128
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only min.6x slower on GPU
    use_crf = True # if crf, training is min.7x slower on CPU
    use_chars = False # if char embedding, training is 3.5x slower on CPU

    # 添加
    dim_pos=50  #词性的维度

    dim_ps=50  # 关系词的维度