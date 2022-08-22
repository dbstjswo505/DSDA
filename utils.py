import numpy as np
import os
import torch
import matplotlib.pylab as plt
import shutil


class ansi:
    BLACK = '\033[30m'
    GRAY = '\033[37m'
    DARKGRAY = '\033[90m'
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    ENDC = '\033[0m'


def save_checkpoint(state, is_best_valid, is_best_test, model_name):

    torch.save(state, os.path.join('params/', model_name + ".pt"))

    if is_best_valid:
        shutil.copyfile(os.path.join('params/', model_name + ".pt"), os.path.join('params/', model_name + "_bvalid.pt"))

    if is_best_test:
        shutil.copyfile(os.path.join('params/', model_name + ".pt"), os.path.join('params/', model_name + "_btest.pt"))



def load_checkpoint(net, optimizer, filename, is_cuda=True, remove_module=False, add_module=False):

    if os.path.isfile(filename):
        checkpoint = torch.load(filename) if is_cuda else torch.load(filename, map_location=lambda storage, loc: storage)
        model_state = net.state_dict()

        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        if remove_module:
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        if add_module:
            state_dict = {'module.' + k: v for k, v in state_dict.items() }

        for k, v in state_dict.items():
            if k in model_state and v.size() == model_state[k].size():
                # print("[INFO] Loading param %s with size %s into model."%(k, ','.join(map(str, model_state[k].size()))))
                pass
            else:
                # print("Size in model is ", v.size(), filename)
                print("[WARNING] Could not load params %s in model." % k)

        pretrained_state = {k: v for k, v in state_dict.items() if
                            k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        net.load_state_dict(model_state)

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("[WARNING] Could not find params file %s." % filename)


def print_results(results):

    has_train = 'train' in results[-1] and len(results[-1]['train']) > 0
    has_valid = 'valid' in results[-1] and len(results[-1]['valid']) > 0
    has_test = 'test' in results[-1] and len(results[-1]['test']) > 0

    labels = [l for l in results[-1][list(results[-1].keys())[0]].keys() if l not in ["loss", "acc", "time"]]

    labels_header = ""
    second_header = ""
    format_header = ""

    for label in labels:
        length = max(9, len(label) + 2)
        spaces = length - len(label)
        label_header = " " * (spaces // 2) + label + " " * (spaces - spaces // 2) + "|"
        labels_header += label_header
        second_header += "-" * (len(label_header) - 1) + "|"
        format_header += "{:>" + str(length//2) + ".2f} {:>" + str(length//2 - (1 - length % 2)) + ".2f}|"

    if len(results) == 1 and len(results[-1]['train']['loss']) == 1:

        valid_header = " Valid loss | Valid err |"
        valid_line =   "------------|-----------|"
        test_header = " Test loss | Test err |"
        test_line =   "-----------|----------|"

        print("Epoch   | Batch | Train loss | Train err |%s%s  Dur       %s\n"
              "--------|-------|------------|-----------|%s%s--------    %s" % (valid_header, test_header, labels_header[:-1], valid_line, test_line, second_header[:-1]))


    train_loss_results = [results[i]["train"]["loss"][-1] for i in range(len(results) - 1)] if has_train else []
    best_train_loss = has_train and (results[-1]["train"]["loss"][-1] <= (np.min(train_loss_results) if len(train_loss_results) > 0 else np.inf))
    train_acc_results = [results[i]["train"]["acc"][-1] for i in range(len(results))] if has_train else []
    best_train_acc = has_train and (results[-1]["train"]["acc"][-1] <= np.min(train_acc_results))

    valid_loss_results = [results[i]["valid"]["loss"][-1] for i in range(len(results) - 1)] if has_valid else []
    best_valid_loss = has_valid and (results[-1]["valid"]["loss"][-1] <= (np.min(valid_loss_results) if len(valid_loss_results) > 0 else np.inf))
    valid_acc_results = [results[i]["valid"]["acc"][-1] for i in range(len(results))] if has_valid else[]
    best_valid_acc = has_valid and (results[-1]["valid"]["acc"][-1] <= np.min(valid_acc_results))

    test_loss_results = [results[i]["test"]["loss"][-1] for i in range(len(results) - 1)] if has_test else []
    best_test_loss = has_test and (results[-1]["test"]["loss"][-1] <= (np.min(test_loss_results) if len(test_loss_results) > 0 else np.inf))
    test_acc_results = [results[i]["test"]["acc"][-1] for i in range(len(results))] if has_test else []
    best_test_acc = has_test and (results[-1]["test"]["acc"][-1] <= np.min(test_acc_results))

    class_accuracies = []
    for label in labels:

        if not has_test:
            class_accuracies.append(results[-1]['train'][label][-1] if has_train else 1.0)

        class_accuracies.append(results[-1]['valid'][label][-1] if has_valid else 1.0)

        if has_test:
            class_accuracies.append(results[-1]['test'][label][-1] if has_test else 1.0)


    valid_format =  " {}{:>10.5f}{} | {}{:>9.4f}{} |"
    test_format =   " {}{:>9.5f}{} | {}{:>8.4f}{} |"

    total_time = np.sum([results[-1][phase]["time"][-1] for phase in ["train", "valid", "test"] if phase in results[-1]])
    total_batch = np.sum([len(results[-1][phase]["loss"]) for phase in ["train", "valid", "test"] if phase in results[-1]])

    print((" {:>6} | {:>5} | {}{:>10.5f}{} | {}{:>9.4f}{} |" + valid_format + test_format + " {:>6.1f}s    " + format_header[:-1]).format(
        len(results), total_batch,
        ansi.BLACK if best_train_loss else ansi.GRAY,
        results[-1]["train"]["loss"][-1] if has_train else -1.0,
        ansi.ENDC,
        ansi.RED if best_train_acc else ansi.GRAY,
        results[-1]["train"]["acc"][-1] if has_train else -1.0,
        ansi.ENDC,
        ansi.GREEN if best_valid_loss else ansi.GRAY,
        results[-1]["valid"]["loss"][-1] if has_valid else -1.0,
        ansi.ENDC,
        ansi.RED if best_valid_acc else ansi.GRAY,
        results[-1]["valid"]["acc"][-1] if has_valid else -1.0,
        ansi.ENDC,
        ansi.GREEN if best_test_loss else ansi.GRAY,
        results[-1]["test"]["loss"][-1] if has_test else -1.0,
        ansi.ENDC,
        ansi.RED if best_test_acc else ansi.GRAY,
        results[-1]["test"]["acc"][-1] if has_test else -1.0,
        ansi.ENDC,
        total_time,
        *class_accuracies), end='\r')

    return best_valid_loss, best_valid_acc
