#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree


# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def instance_segmentation_test(self, net, test_loader, config, num_votes=10, debug=False):    
        """
        Test method for instance segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_semantic_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_semantic_labels]
        self.test_instance_probs = [np.zeros((l.shape[0],)) for l in test_loader.dataset.input_semantic_labels]

        # Test saving path
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
            if not exists(join(test_path, 'grid')):
                makedirs(join(test_path, 'grid'))                
            if not exists(join(test_path, 'potentials')):
                makedirs(join(test_path, 'potentials'))
        else:
            test_path = None

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_semantic_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for semantic_label_value in test_loader.dataset.label_values:
                if semantic_label_value not in test_loader.dataset.ignored_labels:
                    val_semantic_proportions[i] = np.sum([np.sum(semantic_labels == semantic_label_value)
                                                for semantic_labels in test_loader.dataset.validation_semantic_labels])
                    i += 1
        else:
            val_semantic_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        instance_semantic_labels = [-1]
        instance_score = [-1]
        instance_points_num = [-1]
        instance_label_inds = [[]]

        empty_index = [1]
        current_instance_label = empty_index.pop()
        # Start test loop
        with torch.no_grad():
            while True:
                print(last_min)
                print('Initialize workers')
                epoch = 0
                for i, batch in enumerate(test_loader):

                    # New time
                    t = t[-1:]
                    t += [time.time()]

                    if i == 0:
                        print('Done in {:.1f}s'.format(t[1] - t[0]))

                    if 'cuda' in self.device.type:
                        batch.to(self.device)

                    # Forward pass
                    outputs = net(batch, config, epoch)

                    t += [time.time()]

                    # Get probs and labels
                    stacked_probs = softmax(outputs["semantic"]).cpu().detach().numpy()
                    s_points = batch.points[0].cpu().numpy()
                    lengths = batch.lengths[0].cpu().numpy()
                    in_inds = batch.input_inds.cpu().numpy()
                    cloud_inds = batch.cloud_inds.cpu().numpy()

                    # Get Instance labels
                    cluster_mask = np.where(outputs["score"].cpu() > config.test_score_thresh)[0]
                    cluster_inds = [outputs["proposal_clusters"][i] for i in cluster_mask]
                    cluster_semantic_labels = torch.tensor([outputs["proposal_clusters_semantic_labels"][i] for i in cluster_mask])
                    cluster_scores = outputs["score"][cluster_mask].cpu().squeeze(-1)
                    torch.cuda.synchronize(self.device)




                    # Get predictions and labels per sphere
                    # ***************************************

                    i0 = 0
                    for b_i, length in enumerate(lengths):
                        # Get prediction
                        points = s_points[i0:i0 + length]
                        probs = stacked_probs[i0:i0 + length]
                        inds = in_inds[i0:i0 + length]
                        c_i = cloud_inds[b_i]

                        if 0 < test_radius_ratio < 1:
                            mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                            inds = inds[mask]
                            probs = probs[mask]

                        # Update current probs in whole cloud
                        self.test_semantic_probs[c_i][inds] = test_smooth * self.test_semantic_probs[c_i][inds] + (1 - test_smooth) * probs
                        i0 += length
                    
                    # Get instance label predictions
                    # ***************************************
                    if 0 < test_radius_ratio < 1:
                        radius_mask = np.sum(s_points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                    
                
                    # NMS
                    clusters = torch.zeros((len(cluster_scores), len(s_points)), dtype=torch.int, device="cpu")
                    for c in range(len(cluster_inds)):
                        clusters[c][cluster_inds[c]] = 1  #(nProposalClusters, N)                
                    intersection = torch.mm(clusters, clusters.t())
                    cluster_pointnum = clusters.sum(1)
                    c_h = cluster_pointnum.unsqueeze(-1).repeat(1, len(clusters))
                    c_v = cluster_pointnum.unsqueeze(0).repeat(len(clusters), 1)
                    ious = intersection / (c_h + c_v - intersection)
                    sort_scores = cluster_scores.argsort().flip(0)
                    pick = []
                    while len(sort_scores):
                        idx = sort_scores[0]
                        pick.append(idx)
                        iou = ious[idx, sort_scores[1:]]
                        remove_inds = np.where(iou>config.NMS_thresh)[0]+1
                        sort_scores = np.delete(sort_scores, remove_inds)
                        sort_scores = np.delete(sort_scores, 0)
                    pick = np.array(pick)
                    clusters = clusters[pick]
                    cluster_scores = cluster_scores[pick]
                    cluster_semantic_labels = cluster_semantic_labels[pick] 

                    # raidus thresh
                    mask = (clusters*radius_mask).sum(1)>config.center_thresh
                    clusters = clusters[mask]
                    cluster_scores = cluster_scores[mask]
                    cluster_semantic_labels = cluster_semantic_labels[mask] 

                    for c in range(len(clusters)):
                        cluster = clusters[c]
                        score = cluster_scores[c].item()
                        semantic_label = cluster_semantic_labels[c].item()
                        points_num = cluster.sum().item()
                        current_inds = in_inds[cluster>0]
                        overlap = self.test_instance_probs[c_i][current_inds]
                        unique, count = np.unique(overlap[overlap>0], return_counts=True)
                        count_overlap = sorted(zip(count,unique),reverse=True)
                        
                        instance_inds = list(current_inds)
                        if count_overlap:
                            for count, unique in count_overlap:
                                unique = int(unique)
                                if instance_semantic_labels[unique] == semantic_label and c > config.center_thresh:
                                    instance_inds += instance_label_inds[unique]
                                    empty_index.append(unique)
                                    
                        self.test_instance_probs[c_i][instance_inds] = current_instance_label
                        
                        if len(instance_semantic_labels) == current_instance_label: 
                            instance_semantic_labels.append(semantic_label)
                            instance_score.append(score)
                            instance_points_num.append(points_num)
                            instance_label_inds.append(instance_inds)
                        else:
                            instance_semantic_labels[current_instance_label] = semantic_label
                            instance_score[current_instance_label] = score
                            instance_points_num[current_instance_label] = points_num
                            instance_label_inds[current_instance_label] = instance_inds

                        if empty_index:
                            current_instance_label = empty_index.pop()
                        else:
                            current_instance_label = len(instance_semantic_labels)



                    
                    # Average timing
                    t += [time.time()]
                    if i < 2:
                        mean_dt = np.array(t[1:]) - np.array(t[:-1])
                    else:
                        mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                        print(message.format(test_epoch, i,
                                            100 * i / config.validation_size,
                                            1000 * (mean_dt[0]),
                                            1000 * (mean_dt[1]),
                                            1000 * (mean_dt[2])))

                # Update minimum od potentials
                new_min = torch.min(test_loader.dataset.min_potentials)
                print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
                #print([np.mean(pots) for pots in test_loader.dataset.potentials])

                # Save predicted cloud
                if True:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    if test_loader.dataset.set == 'validation':
                        print('\nConfusion on sub clouds')
                        Confs = []
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Insert false columns for ignored labels
                            probs = np.array(self.test_semantic_probs[i], copy=True)
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    probs = np.insert(probs, l_ind, 0, axis=1)

                            # Predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                            # Targets
                            targets = test_loader.dataset.input_semantic_labels[i]

                            # Confs
                            Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        # Rescale with the right number of point per class
                        C *= np.expand_dims(val_semantic_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                        # Compute IoUs
                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print(s + '\n')

                    # Save real IoU once in a while
                    print(np.ceil(new_min))
                    if True:

                        # Project predictions
                        print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                        t1 = time.time()
                        proj_probs = []
                        inst_preds = []
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)

                            # print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                            # print(test_loader.dataset.test_proj[i][:5])

                            # Reproject probs on the evaluations points
                            probs = self.test_semantic_probs[i][test_loader.dataset.test_proj[i], :]
                            save_inst_labels = self.test_instance_probs[i][test_loader.dataset.test_proj[i]]
                            proj_probs += [probs]
                            inst_preds += [save_inst_labels]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Show vote results
                        if True:
                            print('Confusion on full clouds')
                            t1 = time.time()
                            Confs = []
                            for i, file_path in enumerate(test_loader.dataset.files):

                                # Insert false columns for ignored labels
                                for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                    if label_value in test_loader.dataset.ignored_labels:
                                        proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

                                # Get the predicted labels
                                preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                                # Confusion
                                targets = test_loader.dataset.validation_semantic_labels[i]
                                Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                            t2 = time.time()
                            print('Done in {:.1f} s\n'.format(t2 - t1))

                            # Regroup confusions
                            C = np.sum(np.stack(Confs), axis=0)

                            # Remove ignored labels from confusions
                            for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                                if label_value in test_loader.dataset.ignored_labels:
                                    C = np.delete(C, l_ind, axis=0)
                                    C = np.delete(C, l_ind, axis=1)

                            IoUs = IoU_from_confusions(C)
                            mIoU = np.mean(IoUs)
                            s = '{:5.2f} | '.format(100 * mIoU)
                            for IoU in IoUs:
                                s += '{:5.2f} '.format(100 * IoU)
                            print('-' * len(s))
                            print(s)
                            print('-' * len(s) + '\n')

                        # Save predictions
                        print('Saving clouds')
                        t1 = time.time()
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Get file
                            points = test_loader.dataset.load_evaluation_points(file_path)

                            # Get the predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                            # Save plys
                            cloud_name = file_path.split('/')[-1]
                            test_name = join(test_path, 'predictions', cloud_name)
                            write_ply(test_name,
                                    [points, preds, inst_preds[i]],
                                    ['x', 'y', 'z', 'semantic_preds', 'instance_preds'])
                            # test_name2 = join(test_path, 'probs', cloud_name)
                            # prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                            #             for label in test_loader.dataset.label_values]
                            # write_ply(test_name2,
                            #         [points, proj_probs[i], inst_preds[i]],
                            #         ['x', 'y', 'z'] + prob_names +['instance_preds'])
                            preds = test_loader.dataset.label_values[np.argmax(self.test_semantic_probs[i], axis=1)].astype(np.int32)
                            test_name3 = join(test_path, 'grid', cloud_name)
                            write_ply(test_name3,
                                    [points, preds, self.test_instance_probs[i]],
                                    ['x', 'y', 'z', 'semantic_preds', 'instance_preds'])
                            print(f"instance_label_num: {len(np.unique(self.test_instance_probs[i]))}")
                            # Save potentials
                            pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                            pot_name = join(test_path, 'potentials', cloud_name)
                            pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                            write_ply(pot_name,
                                    [pot_points.astype(np.float32), pots],
                                    ['x', 'y', 'z', 'pots'])

                            # Save ascii preds
                            if test_loader.dataset.set == 'test':
                                if test_loader.dataset.name.startswith('Semantic3D'):
                                    ascii_name = join(test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
                                else:
                                    ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                                np.savetxt(ascii_name, preds, fmt='%d')

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                test_epoch += 1

                # Break when reaching number of desired votes
                if last_min > num_votes:
                    break
        return