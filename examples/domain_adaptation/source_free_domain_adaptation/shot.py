import random
import time
import warnings
import argparse
import shutil
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm

import utils
from tllib.alignment.dann import ImageClassifier
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
from tllib.modules.loss import LabelSmoothSoftmaxCEV1
from tllib.modules.entropy import entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_single_dataset(args.data, args.root, args.domain, train_transform, val_transform, args.val_ratio)
    # create indexed dataset in train_target phase
    if args.phase == 'train_target':
        train_dataset = utils.IndexedDataset(train_dataset)
        val_dataset = utils.IndexedDataset(val_dataset)
    # should not drop last in train_target phase
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    args.iters_per_epoch = len(train_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier.bottleneck.apply(init_weights)
    classifier.head.apply(init_weights)

    if args.wn:
        classifier.head = weight_norm(classifier.head)

    if args.load:
        checkpoint = torch.load(args.load, map_location='cpu')
        classifier.load_state_dict(checkpoint)
    
    if not args.load and not args.phase in ['train_source', 'train_target']:
        # resume from the best checkpoint
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # define optimizer and lr scheduler
    parameters = classifier.get_parameters()
    if args.phase == 'train_target':
        # fix the classifier head
        #######################
        parameters.pop(-1)
    
    optimizer = SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1 + args.lr_gamma * x / (args.epochs * args.iters_per_epoch)) ** (-args.lr_decay))

    # analysis the model
    # if args.phase == 'analysis':
    #     # extract features from both domains
    #     feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
    #     source_feature = collect_feature(train_source_loader, feature_extractor, device)
    #     target_feature = collect_feature(train_target_loader, feature_extractor, device)
    #     # plot t-SNE
    #     tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
    #     tsne.visualize(source_feature, target_feature, tSNE_filename)
    #     print("Saving t-SNE to", tSNE_filename)
    #     # calculate A-distance, which is a measure for distribution discrepancy
    #     A_distance = a_distance.calculate(source_feature, target_feature, device)
    #     print("A-distance =", A_distance)
    #     return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        if args.phase == "train_source":
            train_source(train_loader, classifier, optimizer, lr_scheduler, epoch, args)
        else:
            pseudo_labels = collect_pseudo_labels(val_loader, classifier, args)
            train_target(train_loader, pseudo_labels, classifier, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def collect_pseudo_labels(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace):
    model.eval()
    model.training = True # For feature returning
    feature_list = []
    output_list = []
    index_list = []
    with torch.no_grad():
        for data, index in val_loader:
            x, _ = data[:2]
            x = x.to(device)
            y, f = model(x)
            feature_list.append(f.detach().cpu())
            output_list.append(y.detach().cpu())
            index_list.append(index)
    model.training = False
    all_feature = torch.cat(feature_list)
    all_output = torch.cat(output_list)
    all_output = all_output.softmax(dim=1)
    all_index = torch.cat(index_list)
    num_classes = all_output.size(1)

    # Why?
    all_feature = torch.cat((all_feature, torch.ones(all_feature.size(0), 1)), 1)
    all_feature = (all_feature.t() / torch.norm(all_feature, p=2, dim=1)).t()
    
    for _ in range(2):
        centroids = torch.mm(all_output.T, all_feature) / (all_output.sum(dim=0).unsqueeze(1) + args.epsilon) # (num_classes, feature_dim)
        # cosine similarity
        dists = torch.mm(F.normalize(all_feature, dim=1), F.normalize(centroids, dim=1).T) # (num_samples, num_classes)
        pesudo_labels = dists.argmin(dim=1)
        all_output = F.one_hot(pesudo_labels, num_classes).float()
    
    sorted_pesudo_labels = pesudo_labels[torch.argsort(all_index)]
    return sorted_pesudo_labels


def train_source(train_loader: DataLoader, model: ImageClassifier, optimizer: SGD,
                 lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    
    criterion = LabelSmoothSoftmaxCEV1(args.lb_smooth)

    end = time.time()
    for i, data in enumerate(train_loader):
        x, labels = data[:2]
        x, labels = x.to(device), labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, _ = model(x)

        loss = criterion(y, labels)

        cls_acc = accuracy(y, labels)[0]

        losses.update(loss.item(), x.size(0))
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train_target(train_loader: DataLoader, pesudo_labels: torch.LongTensor, model: ImageClassifier,
                 optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    # cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    #######################
    model.head.eval()

    end = time.time()
    for i, (data, index) in enumerate(train_loader):
        # one should never access the target label
        x, pesudo_label = data[0], pesudo_labels[index]
        x, pesudo_label = x.to(device), pesudo_label.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, _ = model(x)
        probs = torch.softmax(y, dim=1)

        entropy_loss = entropy(probs, reduction="mean")
        # Add logK to obtain positive value
        divergence_loss = np.log(probs.size(1)) - entropy(probs.mean(dim=0, keepdim=True), reduction="mean")
        # No label smoothing here
        pesudo_label_loss = F.cross_entropy(y, pesudo_label)

        loss = entropy_loss + divergence_loss + args.trade_off * pesudo_label_loss
        # print(entropy_loss.item(), divergence_loss.item())

        losses.update(loss.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='OfficeHome', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: OfficeHome)')
    parser.add_argument('--domain', help='domain(s) to use', nargs='+')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='the ratio of validation data in the training set for random splitting')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--wn', action='store_true', help='weight normalization for the head layer.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--load', default=None, type=str, help='Directory to load a model. Train only')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=10, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-e', '--epsilon', default=1e-5, type=float, help='threshold for normalization')
    parser.add_argument('--lb-smooth', default=0.1, type=float, help='Ratio of label smoothing. Source only.')
    parser.add_argument('--trade-off', default=0.3, type=float,
                        help='the trade-off hyper-parameter for pesudo-label loss')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='shot',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train_source', choices=['train_source', 'train_target', 'test'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)