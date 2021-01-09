from fastai.vision.all import *
from fastai.callback.fp16 import *

import pandas as pd
import numpy as np

import wandb
from fastai.callback.wandb import *

wandb.init(project="cassava", entity="teo03") #config="config.yaml"

path_str = './data'
PATH = Path(path_str)
images_path = Path(path_str + '/train_images')
csv_path = Path(path_str + '/train.csv')
train_df = pd.read_csv(csv_path)


msk = np.random.rand(len(train_df)) < 0.2
train_df = train_df[msk]

def get_x(row): return images_path/row['image_id']
def get_y(row): return row['label']

blocks = (ImageBlock, CategoryBlock)
splitter = RandomSplitter(valid_pct=0.2)
item_tfms = [Resize(wandb.config.image_size)]
batch_tfms = [RandomResizedCropGPU(280), *aug_transforms(flip_vert=True,
                                                         do_flip=True), Normalize.from_stats(*imagenet_stats)]

block = DataBlock(blocks=blocks, get_x=get_x, get_y=get_y, splitter=splitter, item_tfms=item_tfms, batch_tfms=batch_tfms)
dls = block.dataloaders(train_df, bs=wandb.config.batch_size)

learn = cnn_learner(dls,
                    eval(wandb.config.model),
                    loss_func=LabelSmoothingCrossEntropy(),
                    metrics=accuracy, opt_func=eval(wandb.config.optimizer),
                    cbs=[WandbCallback(), SaveModelCallback()]
                   ).to_fp16()

learn.fine_tune(wandb.config.epochs, base_lr=wandb.config.learning_rate, freeze_epochs=wandb.config.freeze_epochs)



