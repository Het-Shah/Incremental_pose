# """Script for multi-gpu training for incremental learing."""
import json
import os
from copy import deepcopy

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
from alphapose.utils.metrics import DataLogger, calc_accuracy
from transforms_utils import get_max_pred
from animal_data_loader import AnimalDatasetCombined, ToTensor

num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d

def train(opt, train_loader, m, criterion, optimizer, writer, phase="Train"):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()

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
            loss = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))
            acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

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
            board_writing(writer, loss_logger.avg, acc_logger.avg, opt.trainIters, phase)

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

def validate(m, val_loader, opt, cfg, writer, criterion, batch_size=1):
	loss_logger_val = DataLogger()
	acc_logger=DataLogger()

	m.eval()
	
	val_loader = tqdm(val_loader, dynamic_ncols=True)

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

def train_kd(opt, train_loader, m, m_prev, criterion, optimizer, writer, phase = "Train"):
    loss_logger = DataLogger()
    acc_logger = DataLogger()
    m.train()

    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, label_masks) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.cuda().requires_grad_() for inp in inps]
        else:
            inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        label_masks = label_masks.cuda()

        output = m(inps)

        output_teacher = m_prev(inps)

        loss_orig = 0.8 * criterion(output.mul(label_masks), labels.mul(label_masks))

        loss_kd = 0.2 * criterion(output.mul(label_masks), output_teacher.mul(label_masks))

        acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))

        loss = loss_orig + loss_kd

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
            board_writing(writer, loss_logger.avg, acc_logger.avg, opt.trainIters, phase)

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

def main():
	logger.info('******************************')
	logger.info(opt)
	logger.info('******************************')
	logger.info(cfg)
	logger.info('******************************')

	# Model Initialize
	m = preset_model(cfg)
	m = nn.DataParallel(m).cuda()
	m_prev = deepcopy(m)

	if len(cfg.ANIMAL_CLASS_INCREMENTAL) % cfg.INCREMENTAL_STEP != 0:
		print("Number of classes for incremental step is not a multiple of the number of incremental steps!")
		return

	criterion = builder.build_loss(cfg.LOSS).cuda()
	
	if cfg.TRAIN.OPTIMIZER == 'adam':
		optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
	elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
		optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)
	
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)
		
	writer = SummaryWriter('.tensorboard/{}-{}'.format(opt.exp_id, cfg.FILE_NAME))

	# generating base data loaders 
	annot_df = pd.read_csv(cfg.DATASET.ANNOT)

	train_datasets = []
	val_datasets = []

	classes_till_now = []

	for animal_class in cfg.ANIMAL_CLASS_BASE:
		classes_till_now.append(animal_class)
		temp_df = annot_df.loc[annot_df['class']==animal_class]

		images_list = np.array(temp_df['filename'])
		np.random.shuffle(images_list)
		
		train_images_list = images_list[:int(0.05*len(images_list))]
		val_images_list = images_list[int(0.95*len(images_list)):]

		train_tempset = AnimalDatasetCombined(cfg.DATASET.IMAGES, cfg.DATASET.ANNOT, train_images_list, input_size=(512,512), output_size=(128,128), transforms = torchvision.transforms.Compose([ToTensor()]), train=True)

		val_tempset = AnimalDatasetCombined(cfg.DATASET.IMAGES, cfg.DATASET.ANNOT, val_images_list, input_size=(512,512), output_size=(128,128), transforms = torchvision.transforms.Compose([ToTensor()]), train=False)

		train_datasets.append(train_tempset)
		val_datasets.append(val_tempset)
		print(len(train_tempset))

	base_trainset = torch.utils.data.ConcatDataset(train_datasets)
	base_train_loader = torch.utils.data.DataLoader(base_trainset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True)

	base_valset = torch.utils.data.ConcatDataset(val_datasets)
	base_val_loader = torch.utils.data.DataLoader(base_valset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE)


	opt.trainIters = 0
	opt.val_iters = 0

	best_acc = 0.0	
	best_model_weights = deepcopy(m.state_dict())
	logger.info(f'############# Starting Base Training with base classes {cfg.ANIMAL_CLASS_BASE} ########################')

	for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
		opt.epoch = i
		current_lr = optimizer.state_dict()['param_groups'][0]['lr']
		
		logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')
		
		# Training
		
		train_loss, train_acc = train(opt, base_train_loader, m, criterion, optimizer, writer, phase="Base_Train")
		logger.epochInfo('Base_Train', opt.epoch, train_loss, train_acc)
		
		lr_scheduler.step()

		# Prediction Test
		with torch.no_grad():
			val_loss, val_acc = validate(m, base_val_loader, opt, cfg, writer, criterion, batch_size=cfg.TRAIN.VAL_BATCH_SIZE)
			logger.info(f'##### Epoch {opt.epoch} | Loss: {val_loss} | acc: {val_acc} #####')

			# Tensorboard
			if opt.board:
				board_writing(writer, val_loss, val_acc, opt.val_iters, 'Base_Val')

			opt.val_iters += 1

		if val_acc > best_acc: 
			# Save best weights
			best_model_weights = deepcopy(m.state_dict())
			best_acc = val_acc
				
		# Time to add DPG
		if i == cfg.TRAIN.DPG_MILESTONE:
			torch.save(best_model_weights, './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, "Base"))
			# Adjust learning rate
			for param_group in optimizer.param_groups:
				param_group['lr'] = cfg.TRAIN.LR
			
			lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
			
			base_trainset = torch.utils.data.ConcatDataset(train_datasets)
			base_train_loader = torch.utils.data.DataLoader(base_trainset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True)

	torch.save(best_model_weights, './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, "Base"))
	m.load_state_dict(best_model_weights)
	m_prev = deepcopy(m)
	# m_prev.load_state_dict(best_model_weights)
	# m_prev = nn.DataParallel(m_prev)
	# Time to do incremental learning
	train_datasets_incremental = []
	val_datasets_incremental = []

	# train_datasets_incremental.append(base_incremental_train_dataset)
	val_datasets_incremental.append(base_valset)

	for i in range(int(len(cfg.ANIMAL_CLASS_INCREMENTAL)/cfg.INCREMENTAL_STEP)):
		if cfg.TRAIN_INCREMENTAL.OPTIMIZER == 'adam':
			optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
		elif cfg.TRAIN_INCREMENTAL.OPTIMIZER == 'rmsprop':
			optimizer = torch.optim.RMSprop(m.parameters(), lr=cfg.TRAIN.LR)
		
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
		optimizer, milestones=cfg.TRAIN_INCREMENTAL.LR_STEP, gamma=cfg.TRAIN_INCREMENTAL.LR_FACTOR)

		animal_classes = cfg.ANIMAL_CLASS_INCREMENTAL[i*cfg.INCREMENTAL_STEP:(i+1)*cfg.INCREMENTAL_STEP]

		curr_train_datasets = []
		curr_rolling_sets = []
		curr_val_datasets = []

		for animal_class in animal_classes:
			classes_till_now.append(animal_class)
			temp_df = annot_df.loc[annot_df['class']==animal_class]

			images_list = np.array(temp_df['filename'])
			np.random.shuffle(images_list)

			train_images_list = images_list[:int(0.05*len(images_list))]
			val_images_list = images_list[int(0.95*len(images_list)):]

			train_tempset = AnimalDatasetCombined(cfg.DATASET.IMAGES, cfg.DATASET.ANNOT, train_images_list, input_size=(512,512), output_size=(128,128), transforms = torchvision.transforms.Compose([ToTensor()]), train=True)

			val_tempset = AnimalDatasetCombined(cfg.DATASET.IMAGES, cfg.DATASET.ANNOT, val_images_list, input_size=(512,512), output_size=(128,128), transforms = torchvision.transforms.Compose([ToTensor()]), train=False)

			curr_train_datasets.append(train_tempset)
			curr_val_datasets.append(val_tempset)
			print(len(train_tempset))

		samples_per_class = int(cfg.MEMORY / len(classes_till_now))

		for animal_class in classes_till_now:
			temp_df = pd.read_csv(cfg.DATASET.DATA_ROOT+animal_class+'.csv')
			images_list = np.array(temp_df['filename'][:samples_per_class])

			train_tempset = AnimalDatasetCombined(cfg.DATASET.IMAGES, cfg.DATASET.ANNOT, images_list, input_size=(512,512), output_size=(128,128), transforms = torchvision.transforms.Compose([ToTensor()]), train=True)

			curr_train_datasets.append(train_tempset)
			print(len(train_tempset))

		for val_set in curr_val_datasets:
			val_datasets_incremental.append(val_set)

		incremental_trainset = torch.utils.data.ConcatDataset(curr_train_datasets)
		incremental_train_loader = torch.utils.data.DataLoader(incremental_trainset, batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE, shuffle=True)

		incremental_valset = torch.utils.data.ConcatDataset(val_datasets_incremental)
		incremental_val_loader_overall = torch.utils.data.DataLoader(incremental_valset, batch_size=cfg.TRAIN_INCREMENTAL.VAL_BATCH_SIZE)

		incremental_val_loaders_individual = []
		for val_set in val_datasets_incremental: 
			temp_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.TRAIN_INCREMENTAL.VAL_BATCH_SIZE)
			incremental_val_loaders_individual.append(temp_loader)

		opt.trainIters = 0
		opt.val_iters = 0
        
		best_acc = 0.0
		best_model_weights = deepcopy(m.state_dict())

		logger.info(f'#######################################################################################################################')
		logger.info(f'############# Starting Incremental Training step {i} with incremental classes {animal_classes} ########################')

		for ep in range(cfg.TRAIN_INCREMENTAL.BEGIN_EPOCH, cfg.TRAIN_INCREMENTAL.END_EPOCH):
			opt.epoch = ep
			current_lr = optimizer.state_dict()['param_groups'][0]['lr']
			
			logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')
			
			# Training
			if cfg.TRAIN_INCREMENTAL.KD_LOSS: 
				train_loss, train_acc = train_kd(opt, incremental_train_loader, m, m_prev, criterion, optimizer, writer, phase="Incremental_Train"+str(i))
			else:
				train_loss, train_acc = train(opt, incremental_train_loader, m, criterion, optimizer, writer, phase="Incremental_Train"+str(i))
			logger.epochInfo('Incremental_Train'+str(i), opt.epoch, train_loss, train_acc)
			
			lr_scheduler.step()

			# Prediction Test
			with torch.no_grad():
				for class_num in range(len(incremental_val_loaders_individual)):
					val_loss, val_acc = validate(m, incremental_val_loaders_individual[class_num], opt, cfg, writer, criterion, batch_size=cfg.TRAIN_INCREMENTAL.VAL_BATCH_SIZE)
					logger.info(f'##### Evaluating on class {class_num} Epoch {opt.epoch} | Loss: {val_loss} | acc: {val_acc} #####')

				val_loss, val_acc = validate(m, incremental_val_loader_overall, opt, cfg, writer, criterion, batch_size=cfg.TRAIN.VAL_BATCH_SIZE)
				logger.info(f'##### Evaluating on all classes Epoch {opt.epoch} | Loss: {val_loss} | acc: {val_acc} #####')

				if opt.board:
					board_writing(writer, val_loss, val_acc, opt.val_iters, 'Incremental_Val'+str(i))

				opt.val_iters += 1

			if val_acc > best_acc:
				best_acc = val_acc
				best_model_weights = deepcopy(m.state_dict())
				
			# Time to add DPG
			if i == cfg.TRAIN.DPG_MILESTONE:
				torch.save(best_model_weights, './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, 'Incremental'+str(i)))
				# Adjust learning rate
				for param_group in optimizer.param_groups:
					param_group['lr'] = cfg.TRAIN.LR
				
				lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
				
				incremental_trainset = torch.utils.data.ConcatDataset(incremental_train_datasets)
				incremental_train_loader = torch.utils.data.DataLoader(incremental_trainset, batch_size=cfg.TRAIN_INCREMENTAL.BATCH_SIZE * num_gpu, shuffle=True)
		torch.save(best_model_weights, './exp/{}-{}/model_{}.pth'.format(opt.exp_id, cfg.FILE_NAME, 'Incremental'+str(i)))
		m.load_state_dict(best_model_weights)
		m_prev = deepcopy(m)
		m_prev = nn.DataParallel(m_prev).cuda()
				
	torch.save(best_model_weights, './exp/{}-{}/final_weights.pth'.format(opt.exp_id, cfg.FILE_NAME))
	
	

def preset_model(cfg):
	model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
	model._initialize()

	if cfg.MODEL.PRETRAINED:
		logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
		model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
		logger.info(f'Modelloaded!')
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
