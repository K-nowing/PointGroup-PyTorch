from numpy.lib.function_base import append
from models.blocks import *
from scipy.stats import mode
import numpy as np
import time
import multiprocessing
from contextlib import contextmanager


def p2p_fitting_regularizer(net):
    
    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0, no_relu=True)

        #### offset
        # self.offset_head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        # self.offset_head_out = UnaryBlock(config.first_features_dim, 3, False, 0, no_relu=True)
        self.offset_head_mlp = UnaryBlock(out_dim, 3, False, 0)
        

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):
        output = dict()
        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        semantic_output = self.head_mlp(x, batch)
        semantic_output = self.head_softmax(semantic_output, batch)

        offset_output = self.offset_head_mlp(x)
        # offset_output = self.offset_head_out(offset_output)
        
        output["point_features"] = x
        output["semantic"] = semantic_output
        output["offset"] = offset_output
        return output

    def semantic_loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss
    
    def offset_loss(self, outputs, gt_offsets):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param gt_offsets: gt_offsets
        :param inst_num: the number of instance 
        :return: loss
        """
        N = len(outputs)
        pt_diff = outputs - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        self.offset_norm_loss = torch.sum(pt_dist) / N

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        outputs_norm = torch.norm(outputs, p=2, dim=1)
        outputs_ = outputs / (outputs_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * outputs_).sum(-1)   # (N)
        self.offset_dir_loss = torch.sum(direction_diff) / N

        # Combined loss
        return self.offset_dir_loss, self.offset_norm_loss

    def semantic_accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total

########################################################
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        #self.bn4 = nn.BatchNorm1d(512)
        #self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
    
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        #self.bn4 = nn.BatchNorm1d(512)
        #self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.conv2 = torch.nn.Conv1d(channel, 256, 1)
        # self.conv3 = torch.nn.Conv1d(channel, 1024, 1)
        self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=128)

    def forward(self, x):
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
            raise
        else:
            trans_feat = None

        pointfeat = x
        # x = nn.functional.relu(self.bn2(self.conv2(x)))
        # x = self.bn3(self.conv3(x))
        x = self.bn2(self.conv2(x))
        x = torch.mean(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        if self.global_feat:
            return x, trans_feat



def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.max(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class ScoreNet1(nn.Module):
    """
        Get cluster's score
        (clusters_num, points_num, features_num) -> (clusters_num) 
    """
    pass
    def __init__(self, k=1, channel=128):
        super().__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=channel)
        # self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, clusters, features):
        x_concat = []
        trans_concat = []
        for c in clusters:
            x = features[c].T.unsqueeze(0)
            x, trans_feat = self.feat(x)
            x_concat.append(x)
            trans_concat.append(trans_feat)
        x = torch.cat(x_concat, 0)
        #trans_feat = torch.cat(trans_concat, 0)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        # x = nn.functional.relu(self.fc1(x))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x, trans_feat
    
    def loss(self, predict, target):
        score_loss = self.criterion(predict.view(-1), target.view(-1))
        return score_loss.mean()

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = nn.functional.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

#######################################################################################
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device(x.device)#'cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

import torch.nn.functional as F
class ScoreNet(nn.Module):
    def __init__(self, k=20, output_channels=1):
        super().__init__()
        in_c = 128
        self.criterion = nn.BCELoss(reduction='none')

        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(in_c*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(512*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, clusters, features):
        x_concat = []
        for c in clusters:
            x = features[c].T.unsqueeze(0)

            batch_size = x.size(0)
            x = get_graph_feature(x, k=self.k)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            x = get_graph_feature(x1, k=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            x = get_graph_feature(x2, k=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x = get_graph_feature(x3, k=self.k)
            x = self.conv4(x)
            x4 = x.max(dim=-1, keepdim=False)[0]

            x = torch.cat((x1, x2, x3, x4), dim=1)

            x = self.conv5(x)
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            x = torch.cat((x1, x2), 1)
            
            x_concat.append(x)
        x = torch.cat(x_concat, 0)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return torch.sigmoid(x)

    def loss(self, predict, target):
        # print(target.view(-1)[:5])
        # print(predict.view(-1)[:5])
        score_loss = self.criterion(predict.view(-1), target.view(-1))
        return score_loss.mean()
#######################################################################################
class PointGroup(nn.Module):
    def __init__(self, config, lbl_values, ign_lbls):
        super(PointGroup, self).__init__()
        self.backbone = KPFCNN(config, lbl_values, ign_lbls)
        self.scoreNet = ScoreNet()
        self.start_scoring_epoch = 128

        # List of classes ignored during offset shifting
        self.offset_ignored_labels = np.array([0, 1, 2, 3, 4])


    def forward(self, batch, config, epoch):
        outputs = self.backbone(batch, config)

        if epoch > -1:#self.start_scoring_epoch:

            P_points = batch.points[0].clone().detach().cpu()
            Q_points = (batch.points[0]+outputs["offset"]).clone().detach().cpu()

            ### Get P,Q clusters from semantic prediction and offset shift prediction
            semantic_predicts = outputs["semantic"].argmax(-1).clone().detach().cpu()
            neighbors = batch.neighbors[0].clone().detach().cpu()
            # P_clusters = self.clusterSphere(P_points, semantic_predicts, neighbors, r=0.03*3**(1/2))
            # Q_clusters = self.clusterSphere(Q_points, semantic_predicts, neighbors, r=0.03*3**(1/2))
            with poolcontext(processes = 2) as pool:
                (P_clusters, P_semantic_labels), (Q_clusters, Q_semantic_labels) = pool.starmap(PointGroup.clusterSphere, zip([P_points,Q_points], [semantic_predicts]*2, [neighbors]*2))

            outputs["proposal_clusters"] = P_clusters + Q_clusters
            outputs["proposal_clusters_semantic_labels"] = torch.tensor(P_semantic_labels + Q_semantic_labels)
            # outputs["cluster_points"] = P_points + Q_points

            outputs["score"] = self.scoreNet(outputs["proposal_clusters"], outputs["point_features"])

        return outputs

        

    def loss(self, outputs, batch, epoch):
        ## get loss
        loss = {}
        loss["semantic"] = self.backbone.semantic_loss(outputs["semantic"], batch.semantic_labels)
        loss["offset_dir"], loss["offset_reg"] = self.backbone.offset_loss(outputs["offset"], batch.gt_offsets)
        
        self.total_loss = loss["semantic"] + loss["offset_dir"] + loss["offset_reg"]
        loss["score"] = torch.zeros(1)

        if epoch > -1:#self.start_scoring_epoch:

            ### Get target score each proposal cluster
            tg_score = self.getTargetScore(outputs["proposal_clusters"], batch.instance_labels, theta_l=0.25, theta_h=0.75)
            tg_score = tg_score.to(outputs["score"].device)

            loss["score"] = self.scoreNet.loss(outputs["score"], tg_score)
            self.total_loss += loss["score"]
        
        self.loss_dict = loss


        return self.total_loss, self.loss_dict

    def accuracy(self, outputs, batch):
        semantic_accuracy = self.backbone.semantic_accuracy(outputs["semantic"], batch.semantic_labels)
        return semantic_accuracy

    @staticmethod
    def clusterSphere(points, semantic_labels, neighbors, ignore=[0, 1, 2, 3, 4], r=0.04*3**(1/2), points_num_thresh=50):
        """
            points
            semantci_labels
            ignore
        """
        N = len(points)
        visit = [0]*N
        clusters = []
        clusters_semantic_labels = []
        # visit stuff class (e.g ceiling)
        for i in range(N):
            if semantic_labels[i] in ignore:
                visit[i] = 1
                
        for i in range(N):
            if visit[i] == 0:
                queue = list()
                new_cluster = list()

                visit[i] = 1
                queue.append(i)
                new_cluster.append(i)

                while queue:
                    index_1 = queue.pop()
                    point_1 = points[index_1]
                    for index_2 in neighbors[index_1]:
                        index_2 = index_2.item()
                        if index_2 == N:
                            break
                        if visit[index_2] == 1:
                            continue
                        if torch.linalg.norm(point_1 - points[index_2]) >r:
                            break

                        if semantic_labels[index_1] == semantic_labels[index_2]:
                            queue.append(index_2)
                            new_cluster.append(index_2)
                            visit[index_2] = 1

                if len(new_cluster) > points_num_thresh:
                    clusters.append(new_cluster)
                    clusters_semantic_labels.append(semantic_labels[index_1].item())
        return clusters, clusters_semantic_labels
  

    def getTargetScore(self, proposal_clusters, instance_labels, theta_l=0.25, theta_h=0.75):
        """
            porposal_clusters: list of cluster index(list)
            instance_labels
            theta_l
            theta_h
        """
        
        score_list = []
        
        ### get iou
        for p_c in proposal_clusters:
            cluster_label = instance_labels[p_c].cpu()
            mode_value, mode_count = mode(cluster_label)
            if mode_value == -100:
                max_iou = 0
            else:
                max_iou = mode_count.item()/len(p_c)
            score_list.append(max_iou)
        
        ### calculate score and get tensor
        scores = torch.tensor(score_list).float()
        
        fg_mask = scores > theta_h
        bg_mask = scores < theta_l
        interval_inds = (fg_mask == 0) & (bg_mask == 0)

        target_scores = (fg_mask > 0).float()
        k = 1 / (theta_h - theta_l)
        b = theta_l / (theta_l - theta_h)
        target_scores[interval_inds] = scores[interval_inds] * k + b

        return target_scores

