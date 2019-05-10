from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets processing_word=None, processing_pos=None,processing_ps=None,processing_tag=None,
    #                  max_iter=None
    dev   = CoNLLDataset(config.filename_dev,
                         None,
                         config.processing_pos,
                         config.processing_ps,
                         config.processing_tag,
                         config.max_iter)
    train = CoNLLDataset(config.filename_train,
                         None,
                         config.processing_pos,
                         config.processing_ps,
                         config.processing_tag,
                         config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
