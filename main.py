
import torch
import numpy as np
import torchvision.models
import pytorch_lightning as pl
from torchvision import transforms as tfm
from pytorch_metric_learning import losses, miners
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import logging
from os.path import join

import utils
import parser
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from aggregators.gem import GeMPool
from aggregators.cosplace import CosPlace
from aggregators.convap import ConvAP
from aggregators.mixvpr import MixVPR

from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

from pytorch_lightning.callbacks import LearningRateMonitor
import torch.optim.lr_scheduler as lr_scheduler


args = parser.parse_arguments()


class LightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True):
        super().__init__()
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds
        # Use a pretrained model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        # Change the output of the FC layer to the desired descriptors dimension
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)

        if args.aggregator== 'gem':
            print(f"The aggregator used is: {args.aggregator}")
            self.model.avgpool = GeMPool()
        elif args.aggregator == 'cosplace':
            print(f"The aggregator used is: {args.aggregator}")
            self.model.fc = None
            self.model.avgpool = CosPlace(in_dim=512, out_dim=512)
        elif args.aggregator == 'convap':
            print(f"The aggregator used is: {args.aggregator}")
            self.model.fc = None
            self.model.avgpool = ConvAP(in_channels=512)
        elif args.aggregator == 'mixvpr':
            print(f"The aggregator used is: {args.aggregator}")
            #self.model.avgpool = MixVPR(in_channels=128, in_h=28, in_w=28, out_channels=128 , mix_depth=4, mlp_ratio=1, out_rows=4) # we remove layer 3 and 4
            self.model.avgpool = MixVPR(in_channels=256, in_h=14, in_w=14, out_channels=256, mix_depth=4, mlp_ratio=1, out_rows=4) # we remove layer 4
            #self.model.avgpool = MixVPR(in_channels=512, in_h=7, in_w=7, out_channels=512, mix_depth=4, mlp_ratio=1, out_rows=4) # we keep all the layers
            self.model.fc = None # remove fc
            #self.model.layer3 = None #  remove layer3 of resnet18
            self.model.layer4 = None #  remove layer4 of resnet18

        
        # Set the loss function
        if args.loss== 'triplet':
            print(f"The loss used is: {args.loss}")
            self.loss_fn = losses.TripletMarginLoss(margin=0.05, swap=False,smooth_loss=False, triplets_per_anchor="all")
        elif args.loss== 'multisimilarity':
            if args.distance == 'dotproductsimilarity':
                print(f"The loss used is: {args.loss} with distance: {args.distance}")
                self.loss_fn = losses.MultiSimilarityLoss(alpha=args.alpha, beta=args.beta, base=args.base, distance=DotProductSimilarity())
            elif args.distance == 'cosinesimilarity':
                print(f"The loss used is: {args.loss} with distance: {args.distance}")
                self.loss_fn = losses.MultiSimilarityLoss(alpha=args.alpha, beta=args.beta, base=args.base)
        elif args.loss== 'contrastive':
            print(f"The loss used is: {args.loss}")
            self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        
        if args.miner == 'multisimilarityminer':
            print(f"The miner used is: {args.miner}")
            self.miner_fn = miners.MultiSimilarityMiner(epsilon=args.epsilon)
        elif args.miner == 'pairmarginminer':
            print(f"The miner used is: {args.miner}")
            self.miner_fn = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8)
        else: 
            self.miner_fn = None
        

    
    def forward(self, x):
        if args.aggregator == 'avg': 
            descriptors = self.model(x)
            return descriptors
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            if self.model.layer3 is not None:
                x = self.model.layer3(x)
            if self.model.layer4 is not None:
                x = self.model.layer4(x)
            x = self.model.avgpool(x)
            if args.aggregator == 'gem':
                x = self.model.fc(x)
            return x

    
    def configure_optimizers(self):
        if args.optimizer== 'sgd':
            print(f"OPTIMIZER: {args.optimizer} with LEARNING_RATE: {args.learning_rate} and WEIGHT_DECAY: {args.weight_decay}")
            optimizers = torch.optim.SGD(self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer== 'adam':
            print(f"OPTIMIZER: {args.optimizer} with LEARNING_RATE: {args.learning_rate} and WEIGHT_DECAY: {args.weight_decay}")
            optimizers = torch.optim.Adam(self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer== 'adamw':
            print(f"OPTIMIZER: {args.optimizer} with LEARNING_RATE: {args.learning_rate} and WEIGHT_DECAY: {args.weight_decay}")
            optimizers = torch.optim.AdamW(self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer== 'radam':
            print(f"OPTIMIZER: {args.optimizer} with LEARNING_RATE: {args.learning_rate} and WEIGHT_DECAY: {args.weight_decay}")
            optimizers = torch.optim.RAdam(self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer== 'asgd':
            print(f"OPTIMIZER: {args.optimizer} with LEARNING_RATE: {args.learning_rate} and WEIGHT_DECAY: {args.weight_decay}")
            optimizers = torch.optim.ASGD(self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if args.scheduler== 'cosineann':
            print(f"SCHEDULER: {args.scheduler}")
            scheduler = lr_scheduler.CosineAnnealingLR(optimizers, T_max=args.tmax, verbose=True)
            return {
            'optimizer': optimizers,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch', 
                'frequency': 1
                }
            }
        else:
            return optimizers
        
        
        ################## SCHEDULER ##############
        #if args.scheduler== 'reducelr':
        #    print(f"SCHEDULER: {args.scheduler}")
        #    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode='max', patience=2, verbose=True)
        #return {
        #    "optimizer": optimizers,
        #    "lr_scheduler": lr_scheduler,
        #    "monitor": 'val/R@1'
        #}
        #############################################

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        if args.miner == 'none':
            loss = self.loss_fn(descriptors, labels) # no miner
        else:
            miner_output = self.miner_fn(descriptors , labels)
            loss = self.loss_fn(descriptors, labels, miner_output)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        # Feed forward the batch to the model
        descriptors = self(images)  # Here we are calling the method forward that we defined above
        
        loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        if args.scheduler == 'cosineann':
            self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, logger=True)
        return {'loss': loss}

    # For validation and test, we iterate step by step over the validation set
    def inference_step(self, batch):
        images, _ = batch
        descriptors = self(images)
        return descriptors.cpu().numpy().astype(np.float32)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.val_dataset, 'val')

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, 'test', self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, split, num_preds_to_save=0):
        """all_descriptors contains database then queries descriptors"""
        all_descriptors = np.concatenate(all_descriptors)
        queries_descriptors = all_descriptors[inference_dataset.database_num : ]
        database_descriptors = all_descriptors[ : inference_dataset.database_num]

        recalls, recalls_str = utils.compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors,
            self.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )
        # print(recalls_str)
        logging.info(f"Epoch[{self.current_epoch:02d}]): " +
                      f"recalls: {recalls_str}")
    
        self.log(f'{split}/R@1', recalls[0], prog_bar=False, logger=True)
        self.log(f'{split}/R@5', recalls[1], prog_bar=False, logger=True)

def get_datasets_and_dataloaders(args):
    train_transform = tfm.Compose([
        tfm.RandAugment(num_ops=3),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = TrainDataset(
        dataset_folder=args.train_path,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        transform=train_transform
    )
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = parser.parse_arguments()
    utils.setup_logging(join('logs', 'lightning_logs', args.exp_name), console='info')

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args)
    model = LightningModel(val_dataset, test_dataset, args.descriptors_dim, args.num_preds_to_save, args.save_only_wrong_preds)
    
    # Model params saving using Pytorch Lightning. Save the best 3 models according to Recall@1
    checkpoint_cb = ModelCheckpoint(
        monitor='val/R@1',
        filename='_epoch({epoch:02d})_R@1[{val/R@1:.4f}]_R@5[{val/R@5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=1,
        save_last=True,
        mode='max'
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", version=args.exp_name)
    
    ######### SCHEDULER ###############
    #if args.scheduler == 'reducelro':
    #    cb=[checkpoint_cb, LearningRateMonitor()]
    #else:
    #    cb=[checkpoint_cb]
    ###################################
    
    # Instantiate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./logs',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        logger=tb_logger, # log through tensorboard
        callbacks=[checkpoint_cb],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )
    
    #if args.scheduler == 'none':
    trainer.validate(model=model, dataloaders=val_loader, ckpt_path=args.checkpoint)
    #else: 
    trainer.fit(model=model, ckpt_path=args.checkpoint, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path='best')

