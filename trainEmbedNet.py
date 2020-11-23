#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import yaml
import pdb
import glob
import datetime
from utils import *
from EmbedNet import *
from DatasetLoader import get_data_loader
import torchvision.transforms as transforms

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "FaceNet");

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');

## Data loader
parser.add_argument('--batch_size',         type=int, default=200,	help='Batch size, number of classes per batch');
parser.add_argument('--max_img_per_cls',    type=int, default=500,	help='Maximum number of images per class per epoch');
parser.add_argument('--nDataLoaderThread',  type=int, default=5, 	help='Number of loader threads');

## Training details
parser.add_argument('--test_interval',  type=int,   default=5,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=100,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="softmax",  help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.90,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerClass',      type=int,   default=1,      help='Number of images per class per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=8700,   help='Number of classes in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_path',     type=str,   default="data/vggface2", help='Absolute path to the train set');
parser.add_argument('--train_ext',      type=str,   default="jpg",          help='Training files extension');
parser.add_argument('--test_path',      type=str,   default="data/test",    help='Absolute path to the test set');
parser.add_argument('--test_list',      type=str,   default="data/test_list.csv",   help='Evaluation list');

## Model definition
parser.add_argument('--model',          type=str,   default="ResNet18", help='Name of model definition');
parser.add_argument('--nOut',           type=int,   default=512,        help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args();

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Trainer script
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(args):

    ## Load models
    s = EmbedNet(**vars(args)).cuda();

    it          = 1

    ## Write args to scorefile
    scorefile = open(args.result_save_path+"/scores.txt", "a+");

    strtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scorefile.write('%s\n%s\n'%(strtime,args))
    scorefile.flush()

    ## Input transformations for training
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.RandomCrop([224,224]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Input transformations for evaluation
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.CenterCrop([224,224]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(transform=train_transform, **vars(args));
    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    ## If the target directory already exists, start from the existing file
    if len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1]);
        print("Model %s loaded from previous state!"%modelfiles[-1]);
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    elif(args.initial_model != ""):
        trainer.loadParameters(args.initial_model);
        print("Model %s loaded!"%args.initial_model);

    ## If the current iteration is not 1, update the scheduler
    for ii in range(1,it):
        trainer.__scheduler__.step()
    
    ## Evaluation code 
    if args.eval == True:

        sc, lab = trainer.evaluateFromList(transform=test_transform, **vars(args))
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        print('EER %2.4f'%(result[1]))
        quit();

    ## Core training script
    for it in range(it,args.max_epoch+1):

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training epoch %d with LR %f "%(it,max(clr)));

        loss, traineer = trainer.train_network(trainLoader, verbose=True);

        if it % args.test_interval == 0:
            
            sc, lab = trainer.evaluateFromList(transform=test_transform, **vars(args))
            result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

            print("IT %d, VEER %2.4f"%(it, result[1]));
            scorefile.write("IT %d, VEER %2.4f\n"%(it, result[1]));

            trainer.saveParameters(args.model_save_path+"/model%09d.model"%it);

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "TEER/TAcc %2.2f, TLOSS %f"%( traineer, loss));
        scorefile.write("IT %d, TEER/TAcc %2.2f, TLOSS %f\n"%(it, traineer, loss));

        scorefile.flush()

    scorefile.close();


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    if not(os.path.exists(args.model_save_path)):
        os.makedirs(args.model_save_path)
            
    if not(os.path.exists(args.result_save_path)):
        os.makedirs(args.result_save_path)

    main_worker(args)


if __name__ == '__main__':
    main()