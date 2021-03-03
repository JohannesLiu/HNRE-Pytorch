import os
import sys
import pandas as pd

# work_root = "/home/liuxk/PycharmProjects/KGNS"
work_root = "D:/PycharmProjects/KGNS/"

os.chdir(work_root)
sys.path.append("./")
sys.path.append("./plot")

import logging
import torch.nn as nn

from kddirkit.networks.encoders import SentenceEncoder
from kddirkit.networks.models import BaselineModel
from kddirkit.config import *
from kddirkit.dataloaders import WordVec, LoadNYT, LoadHierData
from kddirkit.frameworks import Trainer

from torch import optim

project_name = 'MKATT'

logger = logging.getLogger(project_name)

def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    print('torch.cuda.available:{}'.format(torch.cuda.is_available()))
    print('args["no_cuda"]:{}'.format(args['no_cuda']))
    print("use_cuda:{}".format(use_cuda))
    print("device:{}".format(device))
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    training_batch_size = args['training_batch_size']

    learning_rate = args['learning_rate']

    weight_decay = args['weight_decay']

    keep_prob = args['keep_prob']

    word_size = args['word_size']

    pos_size = args['pos_size']

    hidden_size = args['hidden_size']

    save_file_suffix = str("-tb_" + str(training_batch_size) + "-lr_" + str(learning_rate) + "-weight_decay_"
                        + str(weight_decay) + "-keep_prob_" + str(keep_prob)+ "-pos_size_" + str(pos_size) + "-hidden_size_"+ str(hidden_size))

    HierDataLoader = LoadHierData.HierDataLoader(workdir =os.getcwd(), pattern = "default", device = device)
    relation_levels_Tensor = HierDataLoader.relation_levels_Tensor.to(device)
    relation_level_layer = HierDataLoader.relation_level_layer

    trainDataLoader  = LoadNYT.NYTTrainDataLoader(device = device)
    testDataLoader  = LoadNYT.NYTTestDataLoader(mode = "pr", device = device)
    testDataLoaderPOne  = LoadNYT.NYTTestDataLoader(mode = "pone", device = device)
    testDataLoaderPTwo  = LoadNYT.NYTTestDataLoader(mode = "ptwo", device = device)
    testDataLoaderPAll  = LoadNYT.NYTTestDataLoader(mode = "pall", device = device)

    SkipGramVec = WordVec.SkipGram(data_path = './data/').SkipGramVec
    sentence_encoder = SentenceEncoder.CNNSentenceEncoder(SkipGramVec, 120, args['pos_num'], word_size, pos_size, hidden_size)


    model = BaselineModel.HAttentionNetwork(sentence_encoder=sentence_encoder,
                                            relation_levels=relation_levels_Tensor,
                                            relation_level_layer=relation_level_layer,
                                            keep_prob=0.9,
                                            train_batch_size=training_batch_size,
                                            test_batch_size=args['testing_batch_size'],
                                            num_classes=args['num_classes'],
                                            device=device).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters_to_optimize,
                          learning_rate,
                          weight_decay=weight_decay)

    trainer = Trainer.Trainer(model = model,
                              NYTTrainDataLoader = trainDataLoader,
                              NYTTestDataLoader = testDataLoader,
                              NYTTestDataLoaderPOne= testDataLoaderPOne,
                              NYTTestDataLoaderPTwo = testDataLoaderPTwo,
                              NYTTestDataLoaderPAll= testDataLoaderPAll,
                              args = args,
                              epoch = 0,
                              training_batch_size = training_batch_size,
                              criterion=criterion,
                              optimizer = optimizer,
                              device = device)

    best_auc = 0
    best_result = {}
    for epoch in range(0, args['max_epoch']):
        trainer.train_epoch()
        auc, mi_ma_100, mi_ma_200, pr = trainer.eval_epoch()

        trainer.epoch = trainer.epoch+1

        result = {  "default": auc,
                    "mi_100_10": mi_ma_100['mi_10'],
                    "mi_100_15": mi_ma_100['mi_15'],
                    "mi_100_20": mi_ma_100['mi_20'],
                    "ma_100_10": mi_ma_100['ma_10'],
                    "ma_100_15": mi_ma_100['ma_15'],
                    "ma_100_20": mi_ma_100['ma_20'],
                    "mi_200_10": mi_ma_200['mi_10'],
                    "mi_200_15": mi_ma_200['mi_15'],
                    "mi_200_20": mi_ma_200['mi_20'],
                    "ma_200_10": mi_ma_200['ma_10'],
                    "ma_200_15": mi_ma_200['ma_15'],
                    "ma_200_20": mi_ma_200['ma_20'],
                    "pr-m": pr['m'],
                    "pr-M": pr['M']}

        if best_auc < auc:
            torch.save(trainer.model, args['model_dir'] + args['model'] + "-auc-" + str(auc) + save_file_suffix+ '.pkl')  # save entire net
            torch.save(trainer.model.state_dict(), args['model_dir'] + args['model'] + "-auc-" + str(auc) + "-params" + save_file_suffix+'.pkl')
            best_auc = auc
            best_result = result

        print(result)
        logger.debug('test result ', result)
        logger.debug('Pipe send intermediate result done.')
    print("best result:", best_result)
    logger.debug('Final result is %g', best_result)
    logger.debug('Send final result done.')

def get_params():
    # Training settings
    parser = Parser(work_root + "/data/config", "hnre")
    oneParser = parser.oneParser
    args, _ = oneParser.parse_known_args(args=[])
    return args

if __name__ == '__main__':
    try:
        params = vars(get_params())
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise