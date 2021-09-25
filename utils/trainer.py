import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys

from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config
from sklearn.neighbors import KDTree

from models.blocks import KPConv




class ModelTrainer:
    
    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
                
            else:
                checkpoint = torch.load(chkp_path)
                own_state = net.state_dict()
                for name, param in checkpoint['model_state_dict'].items():
                    if name not in own_state:
                        continue
                    if isinstance(param, Parameter):
                        param = param.data_ptr
                    own_state[name].copy_(param)
                # net.load_state_dict(checkpoint['model_state_dict'])
                # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                # for param in net.backbone.parameters():
                #     param.requires_grad=False
                for param in net.backbone.encoder_blocks.parameters():
                    param.requires_grad=False
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps out_loss offset_loss train_accuracy time\n')

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        # -- tensorboard
        save_dir = './log/encoder_frozen'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger = SummaryWriter(log_dir=save_dir)

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        
        # Start training loop
        for epoch in range(config.max_epoch):
            
            # Remove File for kill signal
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            self.step = 0
            score_loss_list = []
            for batch in training_loader:
                
                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = net(batch, config, epoch)
                total_loss, loss_dict = net.loss(outputs, batch, epoch)
                
                acc = net.accuracy(outputs, batch)

                t += [time.time()]

                # Backward + optimize
                total_loss.backward()

                if config.grad_clip_norm > 0:
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                if not self.device.type == "cpu":
                    torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))
                
                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => Loss:  sem={:.3f} o_dir={:.3f} o_reg={:.3f} c_score={:.3f} | Acc: semantic_acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                    print(message.format(self.epoch, self.step,
                                         loss_dict["semantic"].item(),
                                         loss_dict["offset_dir"].item(),
                                         loss_dict["offset_reg"].item(),
                                         loss_dict["score"].item(),
                                         100*acc,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         1000 * mean_dt[2]))
                
                # logging
                logger.add_scalar("Train/semantic_loss", loss_dict["semantic"].item(), epoch * len(training_loader) + self.step)
                logger.add_scalar("Train/offset_dir_loss", loss_dict["offset_dir"].item(), epoch * len(training_loader) + self.step)
                logger.add_scalar("Train/offset_reg_loss", loss_dict["offset_reg"].item(), epoch * len(training_loader) + self.step)
                logger.add_scalar("Train/score_loss", loss_dict["score"].item(), epoch * len(training_loader) + self.step)
                # Log file
                # if config.saving:
                #     with open(join(config.saving_path, 'training.txt'), "a") as file:
                #         message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                #         file.write(message.format(self.epoch,
                #                                   self.step,
                #                                   net.output_loss,
                #                                   net.reg_loss,
                #                                   acc,
                #                                   t[-1] - t0))


                self.step += 1
                score_loss_list.append(loss_dict["score"].item())
            print("mean_score_loss:", np.mean(score_loss_list))
                

            ##############
            # End of epoch
            ##############

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]

            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            # Validation
            net.eval()
            self.validation(net, val_loader, config)
            net.train()

        print('Finished Training')
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config):

        if config.dataset_task == 'instance_segmentation':
            self.instance_segmentation_validation(net, val_loader, config)
        else:
            raise ValueError('No validation method implemented for this network type')


    def instance_segmentation_validation(self, net, val_loader, config, debug=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############
        
        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Do not validate if dataset has no validation cloud
        if val_loader.dataset.validation_split not in val_loader.dataset.all_splits:
            return

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        #print(nc_tot)
        #print(nc_model)

        # Initiate global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model))
                                     for l in val_loader.dataset.input_semantic_labels]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in val_loader.dataset.label_values:
                if label_value not in val_loader.dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in val_loader.dataset.validation_semantic_labels])
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config, 1)

            # Get probs and labels
            stacked_probs = softmax(outputs["semantic"]).cpu().detach().numpy()
            semantic_labels = batch.semantic_labels.cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            in_inds = batch.input_inds.cpu().numpy()
            cloud_inds = batch.cloud_inds.cpu().numpy()
            if not self.device.type == "cpu":
                torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                target = semantic_labels[i0:i0 + length]
                probs = stacked_probs[i0:i0 + length]
                inds = in_inds[i0:i0 + length]
                c_i = cloud_inds[b_i]

                # Update current probs in whole cloud
                self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                   + (1 - val_smooth) * probs

                # Stack all prediction for this epoch
                predictions.append(probs)
                targets.append(target)
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))
            
        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)


        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)


        t4 = time.time()

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            # Name of saving file
            test_file = join(config.saving_path, 'val_IoUs.txt')

            # Line to write:
            line = ''
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line = line + '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

            # Save potentials
            if val_loader.dataset.use_potentials:
                pot_path = join(config.saving_path, 'potentials')
                if not exists(pot_path):
                    makedirs(pot_path)
                files = val_loader.dataset.files
                for i, file_path in enumerate(files):
                    pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
                    cloud_name = file_path.split('/')[-1]
                    pot_name = join(pot_path, cloud_name)
                    pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
                    write_ply(pot_name,
                            [pot_points.astype(np.float32), pots],
                            ['x', 'y', 'z', 'pots'])

        t6 = time.time()

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))

        # Save predicted cloud occasionally
        if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0:
            val_path = join(config.saving_path, 'val_preds_{:d}'.format(self.epoch + 1))
            if not exists(val_path):
                makedirs(val_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):

                # Get points
                points = val_loader.dataset.load_evaluation_points(file_path)

                # Get probs on our own ply points
                sub_probs = self.validation_probs[i]

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                # Get the predicted labels
                sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                # Reproject preds on the evaluations points
                preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)

                # Path of saved validation file
                cloud_name = file_path.split('/')[-1]
                val_name = join(val_path, cloud_name)

                # Save file
                labels = val_loader.dataset.validation_semantic_labels[i].astype(np.int32)
                write_ply(val_name,
                          [points, preds, labels],
                          ['x', 'y', 'z', 'preds', 'class'])

        # Display timings
        t7 = time.time()
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('Confs bis . {:.1f}s'.format(t4 - t3))
            print('IoU ....... {:.1f}s'.format(t5 - t4))
            print('Save1 ..... {:.1f}s'.format(t6 - t5))
            print('Save2 ..... {:.1f}s'.format(t7 - t6))
            print('\n************************\n')

        return