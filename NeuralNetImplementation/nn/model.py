import numpy as np
import copy
import pickle
import sys
import time
import math

from utils.tools import clip_gradients


class Model():
    def __init__(self, input_shape):
        shape = list(input_shape)
        shape.insert(0, None)
        self.input_shape = tuple(shape)
        self.layers = []
        self.inputs = None
        self.optimizer = None
        self.regularization = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, regularization=None):
        self.optimizer = optimizer
        self.regularization = regularization
        input_shape = self.input_shape
        for layer in self.layers:
            layer.update_output_shape(input_shape)
            input_shape = layer.output_shape
    
    def summary(self):
        print("====== MODEL SUMMARY ======")
        print("input shape:",self.input_shape)
        for layer in self.layers:
            print(layer.name,"output shape:",layer.output_shape)
        print("===========================")
            

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=None, workers=1, use_multiprocessing=False):

        if callbacks!=None:
            print ("WARNING: Model.fit currently does not support callbacks.")
        if class_weight!=None:
            print ("WARNING: Model.fit currently does not support class_weight.")
        if sample_weight!=None:
            print ("WARNING: Model.fit currently does not support sample_weight.")
        if initial_epoch!=0:
            print ("WARNING: Model.fit currently does not support custom initial_epoch. Epoch always starts from 1.")
        if max_queue_size!=None:
            print ("WARNING: Model.fit currently does not support max_queue_size.")
        if use_multiprocessing!=False:
            print ("WARNING: Model.fit currently does not support multiprocessing.")
        if validation_split!=0:
            print ("WARNING: Model.fit currently does not support validation_split. You can provide validation data as (x_val, y_val) tuple in validation_data")

        full_size = x.shape[0]

        if steps_per_epoch != None:
            batch_size = int(full_size*1.0 / steps_per_epoch + 0.5)
        else:
            batch_size = batch_size if batch_size!=None else 32
            steps_per_epoch = math.ceil (full_size*1.0 / batch_size)

        train_results = []
        val_results = []
        for epoch in range(0, epochs, 1):
            
            if(verbose>0):
                print ("==>Epoch:",epoch+1,"/",epochs, " Steps:",steps_per_epoch)
            idxs = np.arange(full_size)
            if shuffle:
                np.random.shuffle(idxs)
            
            # train
            loss_total = 0
            count_correct_total = 0
            for step in range(0, steps_per_epoch, 1):
                iter = epoch*steps_per_epoch+step

                # prepare a batch
                idx_start = step*batch_size
                idx_end = idx_start+batch_size if step<steps_per_epoch-1 else full_size
                idx_batch = idxs[idx_start:idx_end]
                x_batch = x[idx_batch]
                y_batch = y[idx_batch]
                
                # train a batch
                loss, probs = self.forward(x_batch, y_batch)
                self.backward(y_batch)
                self.update(self.optimizer, iter)

                # update stats
                count_correct = np.sum(np.argmax(probs, axis=-1) == y_batch)
                if(verbose>1):
                    print ("Step:",step+1,"/",steps_per_epoch, " loss:",loss," acc:", count_correct*1.0/y_batch.shape[0])
                loss_total += loss
                count_correct_total += count_correct

            # save train stats
            train_loss = loss_total/steps_per_epoch
            train_acc = count_correct_total/full_size
            train_results.append([epoch, train_loss, train_acc])

            val_loss = 'NA'
            val_acc = 'NA'
            if (validation_data!=None) and ((epoch+1)%validation_freq == 0):
                # validate
                x_val = validation_data[0]
                y_val = validation_data[1]
                loss, probs = self.evaluate(x_val, y_val, None, 1, None, validation_steps, None, None, 1, False)

                # save validate stats
                val_loss = loss
                val_acc = np.sum(np.argmax(probs, axis=-1) == y_val)*1.0/y_val.shape[0]
                val_results.append([epoch, val_loss, val_acc])  
            if(verbose>0):    
                print('train_loss:',train_loss,' train_acc:', train_acc, ' val_loss:', val_loss, ' val_acc:', val_acc)

        return np.array(train_results), np.array(val_results)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=None, workers=1, use_multiprocessing=False):
        if callbacks!=None:
            print ("WARNING: Model.fit currently does not support callbacks.")
        if sample_weight!=None:
            print ("WARNING: Model.fit currently does not support sample_weight.")
        if max_queue_size!=None:
            print ("WARNING: Model.fit currently does not support max_queue_size.")
        if use_multiprocessing!=False:
            print ("WARNING: Model.fit currently does not support multiprocessing.")

        # set the mode into testing mode (because dropout layer need this flag)
        for layer in self.layers:
            layer.set_mode(training=False)

        full_size = x.shape[0]
        if steps != None:
            batch_size = int(full_size*1.0 / steps + 0.5)
        else:
            batch_size = batch_size if batch_size!=None else 32
            steps = math.ceil (full_size*1.0 / batch_size)
        
        loss_sum = 0
        probs_all = []
        for step in range(0, steps, 1):
            # prepare a batch
            idx_start = step*batch_size
            idx_end = idx_start+batch_size if step<steps-1 else full_size
            x_batch = x[idx_start:idx_end]
            y_batch = y[idx_start:idx_end]
            loss, probs = self.forward(x_batch, y_batch)
            loss_sum += loss
            probs_all.append(probs)
        
        # reset the mode into training for continous training
        for layer in self.layers:
            layer.set_mode(training=True)

        return loss_sum/steps, np.concatenate(probs_all, axis=0)



    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=None, workers=1, use_multiprocessing=False):
        if callbacks!=None:
            print ("WARNING: Model.predict currently does not support callbacks.")
        if max_queue_size!=None:
            print ("WARNING: Model.predict currently does not support max_queue_size.")
        if use_multiprocessing!=False:
            print ("WARNING: Model.predict currently does not support multiprocessing.")

        # set the mode into testing mode (because dropout layer need this flag)
        for layer in self.layers:
            layer.set_mode(training=False)

        full_size = x.shape[0]
        if steps != None:
            batch_size = int(full_size*1.0 / steps + 0.5)
        else:
            batch_size = batch_size if batch_size!=None else 32
            steps = math.ceil (full_size*1.0 / batch_size)
        
        loss_sum = 0
        probs_all = []
        for step in range(0, steps, 1):
            # prepare a batch
            idx_start = step*batch_size
            idx_end = idx_start+batch_size if step<steps-1 else full_size
            x_batch = x[idx_start:idx_end]
            _, probs = self.forward(x_batch)
            probs_all.append(probs)
        
        # reset the mode into training for continous training
        for layer in self.layers:
            layer.set_mode(training=True)

        return loss_sum/steps, np.concatenate(probs_all, axis=0)


    def forward(self, inputs, targets=None):
        self.inputs = []
        layer_inputs = inputs
        for l, layer in enumerate(self.layers):
            self.inputs.append(layer_inputs)
            if l == len(self.layers)-1:
                layer_inputs, probs = layer.forward(layer_inputs, targets)
            else:
                layer_inputs = layer.forward(layer_inputs)
        outputs = layer_inputs
        return outputs, probs

    def backward(self, targets):
        for l, layer in enumerate(self.layers[::-1]):
            if l == 0:
                grads = layer.backward(self.inputs[-1-l], targets)
            else:
                grads = layer.backward(grads, self.inputs[-1-l])

    def get_params(self):
        params = {}
        grads = {}
        for l, layer in enumerate(self.layers):
            if layer.trainable:
                layer_params, layer_grads = layer.get_params('layer-%dth' % l)
                params.update(layer_params)
                grads.update(layer_grads)

        if self.regularization:
            reg_grads = self.regularization.backward(params)
            for k, v in grads.items():
                grads[k] += reg_grads[k]
        return params, grads

    def update(self, optimizer, iteration):
        params, grads = self.get_params()

        # clip gradients
        # for k, v in grads.items():
        #     grads[k] = clip_gradients(v)
        #     print(k, np.mean(np.abs(v)))

        new_params = optimizer.update(params, grads, iteration)

        for l, layer in enumerate(self.layers):
            if layer.trainable:
                w_key = 'layer-%dth:' % l + layer.name + '/weights'
                b_key = 'layer-%dth:' % l + layer.name + '/bias'
                layer_params = {
                    w_key: new_params[w_key],
                    b_key: new_params[b_key]
                }
                layer.update(layer_params)
