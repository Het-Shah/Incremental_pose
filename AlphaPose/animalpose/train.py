"""Script for multi-gpu training."""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.logger import board_writing, debug_writing
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
# from alphapose.utils.transforms import get_func_heatmap_to_coord
from transforms_utils import get_max_pred
from animal_data_loader import AnimalDatasetCombined, RandomFlip, Noise, ToTensor

num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d

def train(opt, train_loader, m, criterion, optimizer, writer):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)

    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, label_masks) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.cuda().requires_grad_() for inp in inps]
        else:
            inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        label_masks = label_masks.cuda()

        output = m(inps)

        if cfg.LOSS.get('TYPE') == 'MSELoss':
            loss = criterion(output.mul(label_masks), labels.mul(label_masks))
            acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))
        else:
            loss = criterion(output, labels, label_masks)
            acc = calc_integral_accuracy(output, labels, label_masks, output_3d=False, norm_type=norm_type)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        if opt.board:
            board_writing(writer, loss_logger.avg, acc_logger.avg, opt.trainIters, 'Train')

        # Debug
        if opt.debug and not i % 10:
            debug_writing(writer, output, labels, inps, opt.trainIters)

        # TQDM
        train_loader.set_description(
            'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                loss=loss_logger.avg,
                acc=acc_logger.avg)
        )

    train_loader.close()

    return loss_logger.avg, acc_logger.avg

def validate_gt(m, gt_val_loader, opt, cfg, writer, criterion, batch_size=1):
	loss_logger_val = DataLogger()
	acc_logger=DataLogger()

	eval_joints = 16
	
	kpt_json = []
	m.eval()
	
	norm_type = cfg.LOSS.get('NORM_TYPE', None)
	hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
	
	val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

	for inps, labels, label_masks in val_loader:
		if isinstance(inps, list):
			inps = [inp.cuda() for inp in inps]
		
		else:
			inps = inps.cuda()
		labels = labels.cuda()
		label_masks = label_masks.cuda()

		output = m(inps)
		
		loss = criterion(output.mul(label_masks), labels.mul(label_masks))
		acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

		loss_logger_val.update(loss, batch_size)
		acc_logger.update(acc, batch_size)

		# TQDM
		val_loader.set_description(
			'Loss: {loss:.4f} acc: {acc:.4f}'.format(
                	loss=loss_logger_val.avg, acc=acc_logger.avg)
        	)
		
	val_loader.close()
	return loss_logger_val.avg, acc_logger.avg


def main():
	logger.info('******************************')
	logger.info(opt)
	logger.info('******************************')
	logger.info(cfg)
	logger.info('******************************')

    # Model Initialize
	m = preset_model(cfg)
	m = nn.DataParallel(m).cuda()
	
	criterion = builder.build_loss(cfg.LOSS).cuda()
	
	if cfg.TRAIN.OPTIMIZER == 'adam':
		optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
	elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
		optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)
	
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)
		
	writer = SummaryWriter('.tensorboard/{}-{}'.format(opt.exp_id, cfg.FILE_NAME))

	# to make it less random
	annot_df = pd.read_csv(cfg.DATASET.ANNOT)

	if cfg.ANIMAL_CLASS != 'all':
		annot_df = annot_df.loc[annot_df['class']==cfg.ANIMAL_CLASS]
		
	images_list = np.array(annot_df['filename'])
	np.random.shuffle(images_list)

	trainset = AnimalDatasetCombined(cfg.DATASET.IMAGES, cfg.DATASET.ANNOT, images_list, input_size=(512,512), output_size=(128,128), animal_class=cfg.ANIMAL_CLASS, transforms = torchvision.transforms.Compose([ToTensor()]), train=True)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True, num_workers=opt.nThreads)
	
	valset = AnimalDatasetCombined(cfg.DATASET.IMAGES, cfg.DATASET.ANNOT, images_list, input_size=(512,512), output_size=(128,128), animal_class=cfg.ANIMAL_CLASS, transforms = torchvision.transforms.Compose([ToTensor()]), train=False)
	gt_val_loader = torch.utils.data.DataLoader(valset, batch_size=5*num_gpu, shuffle=False, num_workers=opt.nThreads)

	opt.trainIters = 0
	opt.val_iters = 0
	
	for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
		opt.epoch = i
		current_lr = optimizer.state_dict()['param_groups'][0]['lr']
		
		logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')
		
		# Training
		
		loss, miou = train(opt, train_loader, m, criterion, optimizer, writer)
		logger.epochInfo('Train', opt.epoch, loss, miou)
		
		lr_scheduler.step()
		
		if (i + 1) % opt.snapshot == 0:
			# Save checkpoint
			torch.save(m.module.state_dict(), './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
			# Prediction Test
			with torch.no_grad():
				loss, acc = validate_gt(m, gt_val_loader, opt, cfg, writer, criterion, batch_size=5)
				logger.info(f'##### Epoch {opt.epoch} | Loss: {loss} | acc: {acc} #####')

				# Tensorboard
				if opt.board:
					board_writing(writer, loss, acc, opt.val_iters, 'Val')

				opt.val_iters += 1

				#logger.info(f'##### Epoch {opt.epoch} | gt mAP: {gt_AP} #####')
				#rcnn_AP = validate(m.module, opt)
				#logger.info(f'##### Epoch {opt.epoch} | gt mAP: {gt_AP} | rcnn mAP: {rcnn_AP} #####')
				
		# Time to add DPG
		if i == cfg.TRAIN.DPG_MILESTONE:
			torch.save(m.module.state_dict(), './exp/{}-{}/final.pth'.format(opt.exp_id, cfg.FILE_NAME))
			# Adjust learning rate
			for param_group in optimizer.param_groups:
				param_group['lr'] = cfg.TRAIN.LR
			
			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
			
			trainset = AnimalDatasetCombined(cfg.DATASET.IMAGES, cfg.DATASET.ANNOT, images_list, input_size=(512,512), output_size=(128,128), animal_class=cfg.ANIMAL_CLASS, transforms = torchvision.transforms.Compose([ToTensor()]), train=True)
			train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True, num_workers=opt.nThreads)
			
	torch.save(m.module.state_dict(), './exp/{}-{}/final_DPG.pth'.format(opt.exp_id, cfg.FILE_NAME))


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


if __name__ == "__main__":
    main()
