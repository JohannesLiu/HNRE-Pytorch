import sys
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from kddirkit.utils import printer
from kddirkit.metrics import metric
from sklearn.metrics import average_precision_score

class Trainer(object):
    def __init__(self, model, NYTTrainDataLoader, NYTTestDataLoader, NYTTestDataLoaderPOne, NYTTestDataLoaderPTwo, NYTTestDataLoaderPAll, args, epoch, training_batch_size, criterion, optimizer, device, learning_rate=0.2, useScheduler =True, warmup = False,  warmup_step = 1800):
        super(Trainer, self).__init__()
        # load the configArgs
        self.args = args
        self.warmup = warmup
        self.warmup_step = warmup_step
        self.learning_rate = learning_rate

        # load trainning set
        self.NYTTrainDataLoader = NYTTrainDataLoader
        self.train_order = list(range(len(NYTTrainDataLoader.instance_triple)))
        self.train_instance_scope = NYTTrainDataLoader.instance_scope

        # load testing set for pr, hit@100, hit@200
        self.NYTTestDataLoader = NYTTestDataLoader
        self.test_instance_scope = NYTTestDataLoader.instance_scope

        self.exclude_na_flatten_label = self.NYTTestDataLoader.exclude_na_flatten_label
        self.exclude_na_label = np.reshape(self.NYTTestDataLoader.exclude_na_flatten_label, [-1, args['num_classes'] - 1])  # 在这里我们要排除NA标签, 实际上这个文件 all_true_label 就是非出去了第一列NA关系的文件
        self.index_non_zero = np.sum(self.exclude_na_label, 0) > 0

        # load testing set for prone, two, all
        self.NYTTestDataLoaderPOne = NYTTestDataLoaderPOne
        self.test_instance_scopePOne = NYTTestDataLoaderPOne.instance_scope

        # load testing set for prone, two, all
        self.NYTTestDataLoaderPTwo = NYTTestDataLoaderPTwo
        self.test_instance_scopePTwo = NYTTestDataLoaderPTwo.instance_scope

        # load testing set for prone, two, all
        self.NYTTestDataLoaderPAll = NYTTestDataLoaderPAll
        self.test_instance_scopePAll = NYTTestDataLoaderPAll.instance_scope

        #load exclude_na_flatten for px
        self.exclude_na_flatten_labelPX= self.NYTTestDataLoaderPOne.exclude_na_flatten_label
        self.exclude_na_labelPX = np.reshape(self.NYTTestDataLoaderPOne.exclude_na_flatten_label, [-1, args['num_classes'] - 1])  # 在这里我们要排除NA标签, 实际上这个文件 all_true_label 就是非出去了第一列NA关系的文件
        self.index_non_zeroPX = np.sum(self.exclude_na_labelPX, 0) > 0

        # config the model
        self.model = model
        self.training_batch_size = training_batch_size

        self._device = device
        self.loss_history = []
        self.val_acc_history = []

        self.criterion = criterion
        self.optimizer = optimizer
        self.useScheduler = useScheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30000)

        self._epoch = epoch

        # config the eval progress
        self.evalMetric = metric.EvalNYTMetric(self.NYTTestDataLoader.id2rel,
                                               self.NYTTestDataLoader.fewrel_100,
                                               self.NYTTestDataLoader.fewrel_200,
                                               self.exclude_na_flatten_label,
                                               self.exclude_na_label,
                                               self.index_non_zero)
        self.evalMetricX = metric.EvalNYTMetricX(self.exclude_na_flatten_labelPX)

    def train_epoch(self):

        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average losses and metric in this epoch.
        """
        self.model.train()

        cur_lr = self.learning_rate
        if(self.warmup == True):
            cur_lr *= warmup_linear(self.epoch*int(len(self.train_order) / float(self.training_batch_size )), warmup_step = self.warmup_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        # if(self.epoch == 0):
        #     torch.save(self.model, self.args['model_dir'] + self.args['model'] + '-iter-' + str(0) + '.pkl')  # save entire net
        #     torch.save(self.model.state_dict(), self.args['model_dir'] +  self.args['model'] + '-iter-' + str(0) + '.pkl')  # save o
        np.random.shuffle(self.train_order)  # 打乱训练集
        s1 = 0.0
        s2 = 0.0
        tot1 = 0.0
        tot2 = 0.0
        loss_sum = 0.0
        step_sum = 0.0
        for i in range(int(len(self.train_order) / float(self.training_batch_size ))):
            input_scope = np.take(self.train_instance_scope, self.train_order[i * self.training_batch_size :(i + 1) * self.training_batch_size ],  axis=0)
            index = []
            scope = [0]
            label = []
            for num in input_scope:
                index = index + list(range(num[0], num[1] + 1))  # 放入每一个袋子的起点和重点的每个序号
                label.append(self.NYTTrainDataLoader.label[num[0]])  # 这个是每个袋子的第一个数的序号的标签
                scope.append(scope[len(scope) - 1] + num[1] - num[
                    0] + 1)  # 这个存的是，当前的scope 类似于[0,1,2, , 19, 27, 35]， 代表着每一个袋子包含的实例的个数
            #         print("index length:", len(index), "label length", len(label), "scope length", len(scope), "label_", len(label_))
            label_ = np.zeros((self.training_batch_size , self.args['num_classes']))  # 相当于 做了一个onehot_dict
            label_[np.arange(self.training_batch_size ), label] = 1  # 为onehot_dict 赋值
            #         output, losses, correct_predictions = train_step(train_word[index,:], train_pos1[index,:], train_pos2[index,:],
            #             train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope))

            feed_dict = {
                'word': self.NYTTrainDataLoader.word_Tensor[index, :],
                'pos1': self.NYTTrainDataLoader.pos1_Tensor[index, :],
                'pos2': self.NYTTrainDataLoader.pos2_Tensor[index, :],
                'mask': self.NYTTrainDataLoader.mask_Tensor[index, :],
                'len': self.NYTTrainDataLoader.len_Tensor[index],
                'label_index': self.NYTTrainDataLoader.label_Tensor[index],
                'label_': torch.LongTensor(label_).to(self._device),  # 可以不用
                'scope': scope
            }


            logits = self.model(feed_dict)
            output = F.softmax(logits, dim = -1)
            self.optimizer.zero_grad()
            loss = self.criterion(logits, torch.LongTensor(label).to(self._device))  # 计算损失值
            # loss = self.criterion(logits, torch.LongTensor(label_).to(self._device))  # 计算损失值
            loss.backward()  # 反向传播计算参数的梯度
            self.optimizer.step()  # 使用优化方法进行梯度更新
            if self.useScheduler == True:
                self.scheduler.step()
            predictions = torch.argmax(logits, 1)
            correct_predictions = torch.eq(predictions, torch.argmax(torch.LongTensor(label_).to(self._device), 1))
            accuracy = torch.mean(correct_predictions.float())
            num = 0
            s = 0
            for num in correct_predictions:  # 在这160个batch中
                if label[s] == 0:  # 如果预测==0
                    tot1 += 1.0
                    if num:  ##如果==0且正确
                        s1 += 1.0
                else:
                    tot2 += 1.0
                    if num:  ##如果预测！=0且正确
                        s2 += 1.0
                s = s + 1
            loss_sum += loss.item()  # 更新 losses
            step_sum += 1.0  # 更新一个batch 中的step_sum， 疑似等于 i， 这个可能用在外循环上
            if tot1 == 0:  # 防止分母为0, 这里是NA = s1/tot1 , NA的准确率为， 预测正确的NA的数量/NA的总数
                tot1 += 1
            if tot2 == 0:  # 防止分母为0，这里是not NA = s2/tot2, 这里表示，预测正确的非NA的综述, 算是TP查准率
                tot2 += 1

            time_str = datetime.datetime.now().isoformat().replace('T', ' ')
            temp_str = 'epoch {0:0>3}/{1:0>3} step {2:0>4} time {3:26} | losses : {4:1.8f} | NA accuracy: {5:1.6f} | not NA accuracy: {6:1.6f}\r'.format(
                self.args['restore_epoch'] + self.epoch + 1, self.args['restore_epoch'] + self.args['max_epoch'], i, time_str, loss.item(),
                s1 / tot1, s2 / tot2)
            # sys.stdout.write(temp_str)
            # sys.stdout.flush()

        losses = loss_sum / step_sum
        na_acc = s1 / tot1
        not_na_acc = s2 / tot2
        temp_str = 'epoch {0:0>3}/{1:0>3} step {2:0>4} time {3:26} | losses : {4:1.8f} | NA accuracy: {5:1.6f} | not NA accuracy: {6:1.6f}\r'.format(
            self.args['restore_epoch'] + self.epoch + 1, self.args['restore_epoch'] + self.args['max_epoch'], i, time_str, losses,
            na_acc, not_na_acc)
        print(temp_str)

        #     current_step = tf.train.global_step(sess, global_step) #tensorflow 版本的模型保存
        # if (self._epoch + 1) % self.args['save_epoch'] == 0:  # 如果到达一个保存周期
        #     # 2 ways to save the net
        #     torch.save(self.model, self.args['model_dir'] + self.args['model'] + '-epoch-' + str(self.epoch+1) + '.pkl')  # save entire net
        #     torch.save(self.model.state_dict(), self.args['model_dir'] + self.args['model'] + '-epoch-' + str(self.epoch+1) + '.pkl')  # save o
        return losses, na_acc, not_na_acc

    def eval_epoch(self):
        self.model.eval()
        with torch.no_grad():
            stack_output = []  # stack_out 干什么的

            iteration = len(self.test_instance_scope) // self.args['testing_batch_size']
            for i in range(iteration):  # 循环迭代次数
                input_scope = self.test_instance_scope[i * self.args['testing_batch_size']:(i + 1) * self.args['testing_batch_size']]
                index = []
                scope = [0]
                label = []
                for num in input_scope:
                    index = index + list(range(num[0], num[1] + 1))
                    label.append(self.NYTTestDataLoader.label[num[0]])
                    scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
                label_ = np.zeros((self.args['testing_batch_size'], self.args['num_classes']))
                label_[np.arange(self.args['testing_batch_size']), label] = 1
                feed_dict = {
                    'word': self.NYTTestDataLoader.word_Tensor[index, :],
                    'pos1': self.NYTTestDataLoader.pos1_Tensor[index, :],
                    'pos2': self.NYTTestDataLoader.pos2_Tensor[index, :],
                    'mask': self.NYTTestDataLoader.mask_Tensor[index, :],
                    'len': self.NYTTestDataLoader.len_Tensor[index],
                    'label_': label_,  # 可以不用
                    'scope': scope
                }
                output = self.model.forward_infer(feed_dict).cpu().numpy()
                stack_output.append(output)  # 将输出，拼接到输出stack里边

            stack_output = np.concatenate(stack_output, axis=0)  # 拼接输出
            exclude_na_output = stack_output[:, 1:]  # 拼接从排除NA列的输出
            exclude_na_flatten_output = np.reshape(stack_output[:, 1:], (-1))  # 重置stack_output的维度

            auc = average_precision_score(self.exclude_na_flatten_label, exclude_na_flatten_output)
            mi_ma_100 = self.evalMetric.mi_ma_100(exclude_na_output)
            mi_ma_200 = self.evalMetric.mi_ma_200(exclude_na_output)
            pr = self.evalMetric.pr(exclude_na_output, exclude_na_flatten_output)
            return auc, mi_ma_100, mi_ma_200, pr, exclude_na_flatten_output

    def eval_epoch_px(self, mode = "pone"):
        if mode == "pone":
            NYTTestDataLoaderX = self.NYTTestDataLoaderPOne
            test_instance_scopeX = self.test_instance_scopePOne
        elif mode == "pwo":
            NYTTestDataLoaderX = self.NYTTestDataLoaderPTwo
            test_instance_scopeX = self.test_instance_scopePTwo
        elif mode =="pall":
            NYTTestDataLoaderX = self.NYTTestDataLoaderPAll
            test_instance_scopeX = self.test_instance_scopePAll

        self.model.eval()
        with torch.no_grad():
            stack_output = []  # stack_out 干什么的

            iteration = len(test_instance_scopeX) // self.args['testing_batch_size']
            for i in range(iteration):  # 循环迭代次数
                input_scope = test_instance_scopeX[i * self.args['testing_batch_size']:(i + 1) * self.args['testing_batch_size']]
                index = []
                scope = [0]
                label = []
                for num in input_scope:
                    index = index + list(range(num[0], num[1] + 1))
                    label.append(NYTTestDataLoaderX.label[num[0]])
                    scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
                label_ = np.zeros((self.args['testing_batch_size'], self.args['num_classes']))
                label_[np.arange(self.args['testing_batch_size']), label] = 1
                feed_dict = {
                    'word': NYTTestDataLoaderX.word_Tensor[index, :],
                    'pos1': NYTTestDataLoaderX.pos1_Tensor[index, :],
                    'pos2': NYTTestDataLoaderX.pos2_Tensor[index, :],
                    'mask': NYTTestDataLoaderX.mask_Tensor[index, :],
                    'len': NYTTestDataLoaderX.len_Tensor[index],
                    'label_': label_,  # 可以不用
                    'scope': scope
                }
                output = self.model.forward_infer(feed_dict).cpu().numpy()
                stack_output.append(output)  # 将输出，拼接到输出stack里边

            stack_output = np.concatenate(stack_output, axis=0)  # 拼接输出
            exclude_na_flatten_output = np.reshape(stack_output[:, 1:], (-1))  # 重置stack_output的维度
            return self.evalMetricX.pone_two_all(exclude_na_flatten_output)

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch
        return epoch

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
