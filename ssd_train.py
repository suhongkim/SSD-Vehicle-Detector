import time
from time import gmtime, strftime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from ssd_net import SSD
from bbox_loss import MultiboxLoss
from bbox_helper import loc2bbox


def train_net(train_loader, valid_loader, class_labels, lab_results_dir, learning_rate=0.0001, is_lr_scheduled=True, max_epoch=1, save_epochs=[10, 20, 30]):
    # Measure execution time
    train_start = time.time()
    start_time = strftime('SSD__%dth_%H:%M_', gmtime())

    # Define the Net
    print('num_class: ', len(class_labels))
    print('class_labels: ', class_labels)
    ssd_net = SSD(len(class_labels))
    # Set the parameter defined in the net to GPU
    net = ssd_net

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.benchmark = True
        net.cuda()

    # Define the loss
    center_var = 0.1
    size_var = 0.2
    criterion = MultiboxLoss([center_var, center_var, size_var, size_var],
                             iou_threshold=0.5, neg_pos_ratio=3.0)

    # Define Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9,
    #                       weight_decay=0.0005)
    if is_lr_scheduled:
        scheduler = MultiStepLR(optimizer, milestones=[8, 15, 30, 45], gamma=0.1)

    # Train data
    conf_losses = []
    loc_losses = []
    v_conf_losses = []
    v_loc_losses = []
    itr = 0
    train_log = []
    valid_log = []
    last_saved_epoch = 0
    count = 0
    min_loss_avg = 0
    for epoch_idx in range(0, max_epoch):

        # decrease learning rate
        if is_lr_scheduled:
            scheduler.step()
            print('\n\n===> lr: {}'.format(scheduler.get_lr()[0]))

        if epoch_idx >= (last_saved_epoch + 5 + count):
            last_saved_epoch = epoch_idx
            count += 2
            if learning_rate >= 0.00000001:
                learning_rate = learning_rate * 0.1
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
                print(last_saved_epoch, epoch_idx, learning_rate)


        # iterate the mini-batches:
        for train_batch_idx, data in enumerate(train_loader):
            train_images, train_labels, train_bboxes, prior_bbox = data

            # Switch to train model
            net.train()

            # Forward
            train_img = Variable(train_images.clone().cuda())
            train_bbox = Variable(train_bboxes.clone().cuda())
            train_label = Variable(train_labels.clone().cuda())

            train_out_confs, train_out_locs = net.forward(train_img)
            # locations(feature map base) -> bbox(center form)
            train_out_bbox = loc2bbox(train_out_locs, prior_bbox[0].unsqueeze(0))

            # update the parameter gradients as zero
            optimizer.zero_grad()

            # Compute the loss
            conf_loss, loc_loss = criterion.forward(train_out_confs, train_out_bbox, train_label, train_bbox)
            train_loss = conf_loss + loc_loss

            # Do the backward to compute the gradient flow
            train_loss.backward()

            # Update the parameters
            optimizer.step()

            conf_losses.append((itr, conf_loss))
            loc_losses.append((itr, loc_loss))

            itr += 1
            if train_batch_idx % 20 == 0:
                train_log_temp = '[Train]epoch: %d itr: %d Conf Loss: %.4f Loc Loss: %.4f' % (epoch_idx, itr, conf_loss, loc_loss)
                train_log += (train_log_temp + '\n')
                print(train_log_temp)
                if False: # check input tensor
                    image_s = train_images[0, :, :, :].cpu().numpy().astype(
                        np.float32).transpose().copy()  # c , h, w -> h, w, c
                    image_s = ((image_s + 1) / 2)
                    bbox_cr_s = torch.cat([train_bboxes[..., :2] - train_bboxes[..., 2:] / 2,
                                           train_bboxes[..., :2] + train_bboxes[..., 2:] / 2], dim=-1)
                    bbox_prior_s = bbox_cr_s[0, :].cpu().numpy().astype(np.float32).reshape((-1, 4)).copy()  # First sample in batch
                    bbox_prior_s = (bbox_prior_s * 300)
                    label_prior_s = train_labels[0, :].cpu().numpy().astype(np.float32).copy()
                    bbox_s = bbox_prior_s[label_prior_s > 0]
                    label_s = (label_prior_s[label_prior_s > 0]).astype(np.uint8)

                    for idx in range(0, len(label_s)):
                        cv2.rectangle(image_s, (bbox_s[idx][0], bbox_s[idx][1]), (bbox_s[idx][2], bbox_s[idx][3]),
                                      (255, 0, 0), 2)
                        cv2.putText(image_s, class_labels[label_s[idx]], (bbox_s[idx][0], bbox_s[idx][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0),
                                    1, cv2.LINE_AA)

                    plt.imshow(image_s)
                    plt.show()

        # validaton

        if epoch_idx >= 0: #train_batch_idx % 200 == 0:
            net.eval() # Evaluation mode
            v_conf_subsum = torch.zeros(1)  # collect the validation losses for avg.
            v_loc_subsum = torch.zeros(1)
            v_itr = 0
            # v_itr_max = 5
            for valid_itr, data in enumerate(valid_loader):
                valid_image, valid_label, valid_bbox, prior_bbox = data
                valid_image = Variable(valid_image.cuda())
                valid_bbox = Variable(valid_bbox.cuda())
                valid_label = Variable(valid_label.cuda())

                # Forward and compute loss
                with torch.no_grad(): # make all grad flags to false!! ( Memory decrease)
                    valid_out_confs, valid_out_locs = net.forward(valid_image)
                valid_out_bbox = loc2bbox(valid_out_locs, prior_bbox[0].unsqueeze(0))  # loc -> bbox(center form)

                valid_conf_loss, valid_loc_loss = criterion.forward(valid_out_confs, valid_out_bbox, valid_label, valid_bbox)

                v_conf_subsum += valid_conf_loss
                v_loc_subsum += valid_loc_loss
                v_itr += 1
                # valid_itr += 1
                # if valid_itr > v_itr_max:
                #     break

            # avg. valid loss
            v_conf_loss_avg = v_conf_subsum / v_itr
            v_loc_loss_avg = v_loc_subsum / v_itr
            v_conf_losses.append((itr, v_conf_loss_avg))
            v_loc_losses.append((itr, v_loc_loss_avg))

            valid_log_temp = '==>[Valid]epoch: %d itr: %d Conf Loss: %.4f Loc Loss: %.4f' % (
                epoch_idx, itr, v_conf_loss_avg, v_loc_loss_avg)
            valid_log += (valid_log_temp + '\n')
            print(valid_log_temp)


            if min_loss_avg == 0 or v_conf_loss_avg <= min_loss_avg:
                min_loss_avg = v_conf_loss_avg
                last_saved_epoch = epoch_idx
                temp_file = start_time + 'best'
                net_state = net.state_dict()  # serialize the instance
                torch.save(net_state, lab_results_dir + temp_file + '_model.pth')

                temp_data = {'class_labels': class_labels,
                             'conf_losses': np.asarray(conf_losses),
                             'loc_losses': np.asarray(loc_losses),
                             'v_conf_losses': np.asarray(v_conf_losses),
                             'v_loc_losses': np.asarray(v_loc_losses),
                             'learning_rate': learning_rate,
                             'total_itr': itr,
                             'max_epoch': max_epoch,
                             'train_time': start_time
                             }
                torch.save(temp_data, lab_results_dir + temp_file + '.loss')
                print('================> Best data file is created: ', lab_results_dir + temp_file + '_model.pth')





        # Save the trained network
        if epoch_idx in save_epochs:
            temp_file = start_time + 'epoch_{}'.format(epoch_idx)
            net_state = net.state_dict()  # serialize the instance
            torch.save(net_state, lab_results_dir + temp_file + '__model.pth')
            print('================> Temp Model file is created: ', lab_results_dir + temp_file + '__model.pth')

            temp_data = {'class_labels': class_labels,
                         'conf_losses': np.asarray(conf_losses),
                         'loc_losses': np.asarray(loc_losses),
                         'v_conf_losses': np.asarray(v_conf_losses),
                         'v_loc_losses': np.asarray(v_loc_losses),
                         'learning_rate': learning_rate,
                         'total_itr': itr,
                         'max_epoch': max_epoch,
                         'train_time': start_time
                         }
            torch.save(temp_data, lab_results_dir + temp_file + '.loss')
            print('================> Temp data file is created: ', lab_results_dir + temp_file + '.loss')

    # Measure the time
    train_end = time.time()
    m, s = divmod(train_end - train_start, 60)
    h, m = divmod(m, 60)

    # Save the result
    results_file_name = start_time + '%d:%02d:%02d' % (h, m, s)

    train_data = {'class_labels': class_labels,
                  'conf_losses': np.asarray(conf_losses),
                  'loc_losses': np.asarray(loc_losses),
                  'v_conf_losses': np.asarray(v_conf_losses),
                  'v_loc_losses': np.asarray(v_loc_losses),
                  'learning_rate': learning_rate,
                  'total_itr': itr,
                  'max_epoch': max_epoch,
                  'train_time': '%d:%02d:%02d' % (h, m, s)}

    torch.save(train_data, lab_results_dir + results_file_name + '.loss')

    # Save the trained network
    net_state = net.state_dict()  # serialize the instance
    torch.save(net_state,  lab_results_dir + results_file_name + '__model.pth')

    # Save the train/valid log
    torch.save({'log': train_log}, lab_results_dir + results_file_name + '__train.log')
    torch.save({'log': valid_log}, lab_results_dir + results_file_name + '__valid.log')

    return lab_results_dir + results_file_name