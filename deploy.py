import time
import torch
import torch.nn.functional as F
import numpy as np
import pdb
from torch import nn
from torch.autograd import Variable
from utils import print_results, save_checkpoint

def confusion_matrix(l, o):
    matrix = np.zeros((5,5))
    L = len(l)
    for i in range(L):
        label = int(l[i])
        pred = int(o[i])
        matrix[label][pred] = matrix[label][pred] + 1
    return matrix

def test(net, dataloaders, model_name, optimizer, criterion, phases=["test"], max_epochs=1, classlabels=None):

    results = []
    for epoch in range(max_epochs):

        results.append(dict())

        # Each epoch has a training and validation phase
        for phase in phases:

            since = time.time()

            net.eval()

            running_loss = 0.
            num_correct = 0.
            total_samples = 0.
            correct = []

            o = np.zeros((0,))
            l = np.zeros((0,))

            results[-1][phase] = dict(loss=[], time=[], acc=[])
            if classlabels is not None:
                for label in classlabels:
                    results[-1][phase][label] = []

            # Iterate over data.
            for idx, data in enumerate(dataloaders[phase]):

                inputs, labels = data

                if phase != 'train':
                    with torch.no_grad():
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()


                outputs = net(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    z=1
                    #loss.backward()
                    #optimizer.step()

                # statistics
                running_loss += loss.item()

                num_correct += outputs[:labels.size(0)].max(1)[1].eq(labels).sum().item()
                correct = correct + outputs[:labels.size(0)].max(1)[1].eq(labels).tolist()
                total_samples += len(outputs)


                o = np.concatenate((o, outputs[:labels.size(0)].max(1)[1].cpu().data.numpy()))
                l = np.concatenate((l, labels.cpu().data.numpy()))

                del inputs, outputs, labels, loss

                results[-1][phase]["loss"].append(running_loss / (idx + 1))
                results[-1][phase]["acc"].append(1 - num_correct / total_samples)
                results[-1][phase]["time"].append(time.time() - since)

                if classlabels is not None:

                    for label in classlabels:
                        idx = classlabels.index(label)
                        if np.sum(l == idx) > 0:
                            results[-1][phase][label].append(1. - float(np.sum(l[o == idx] == idx)) / np.sum(l == idx))
                        else:
                            results[-1][phase][label].append(1.0)

                best_valid_loss, best_test_loss = print_results(results)
            
            #if best_valid_loss and epoch>30:
            #    matrix = confusion_matrix(l,o)

    print(epoch+1,
          results[epoch]['train']['loss'][-1],
          results[epoch]['valid']['loss'][-1],
          results[epoch]['valid']['acc'][-1],
          results[epoch]['test']['loss'][-1] if 'test' in phases else "",
          results[epoch]['test']['acc'][-1] if 'test' in phases else "" )
    pdb.set_trace()
    tmp = np.array(correct)
    np.save('test_result_a', tmp)   
    z=1

def train(net, dataloaders, model_name, optimizer, criterion, phases=["train", "valid", "test"], max_epochs=1000, classlabels=None):

    assert "train" in phases

    results = []
    for epoch in range(max_epochs):

        results.append(dict())

        # Each epoch has a training and validation phase
        for phase in phases:

            since = time.time()

            if phase == 'train':
                net.train(True)
            else:
                net.eval()

            running_loss = 0.
            num_correct = 0.
            total_samples = 0.

            o = np.zeros((0,))
            l = np.zeros((0,))

            results[-1][phase] = dict(loss=[], time=[], acc=[])
            if classlabels is not None:
                for label in classlabels:
                    results[-1][phase][label] = []

            # Iterate over data.
            for idx, data in enumerate(dataloaders[phase]):

                inputs, labels = data

                if phase != 'train':
                    with torch.no_grad():
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()


                outputs = net(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()

                num_correct += outputs[:labels.size(0)].max(1)[1].eq(labels).sum().item()
                total_samples += len(outputs)


                o = np.concatenate((o, outputs[:labels.size(0)].max(1)[1].cpu().data.numpy()))
                l = np.concatenate((l, labels.cpu().data.numpy()))

                del inputs, outputs, labels, loss

                results[-1][phase]["loss"].append(running_loss / (idx + 1))
                results[-1][phase]["acc"].append(1 - num_correct / total_samples)
                results[-1][phase]["time"].append(time.time() - since)

                if classlabels is not None:

                    for label in classlabels:
                        idx = classlabels.index(label)
                        if np.sum(l == idx) > 0:
                            results[-1][phase][label].append(1. - float(np.sum(l[o == idx] == idx)) / np.sum(l == idx))
                        else:
                            results[-1][phase][label].append(1.0)

                best_valid_loss, best_test_loss = print_results(results)
            
            #if best_valid_loss and epoch>30:
            #    matrix = confusion_matrix(l,o)
        print()
        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': net.state_dict(),
            #'best_loss': results["valid"]["loss"][-1],
            'optimizer': optimizer.state_dict(),
        }, best_valid_loss, best_test_loss, model_name)

    print()
    epoch = np.argmin([results[i]["valid"]["loss"][-1] for i in range(len(results))])

    print(epoch+1,
          results[epoch]['train']['loss'][-1],
          results[epoch]['valid']['loss'][-1],
          results[epoch]['valid']['acc'][-1],
          results[epoch]['test']['loss'][-1] if 'test' in phases else "",
          results[epoch]['test']['acc'][-1] if 'test' in phases else "" )
