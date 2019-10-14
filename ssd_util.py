import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_dataset_list_original(sample_path, label_path, cls_labels):
    # load the each image path
    image_paths = []
    for root, dirs, files in os.walk(sample_path):
        for filename in files:
            image_paths.append(root + '/' + filename)

    # load the each label path
    polygon_paths = []
    for root, dirs, files in os.walk(label_path):
        for filename in files:
            if filename.find('.json') > 0:
                polygon_paths.append(root + '/' + filename)

    # extract bounding box
    dataset_list = []
    for image_path, polygon_dir in zip(image_paths, polygon_paths):
        with open(polygon_dir, 'r') as p:
            polygon = json.load(p)
        # print([obj['label'] for obj in polygon['objects']])
        bbox_list = []
        cls_list = []
        lbl_list = []
        for obj in polygon['objects']:
            # cls = [0, 0, 0, 0]
            if obj['label'] in cls_labels:
                # cls[cls_labels.index(obj['label'])] = 1
                cls = cls_labels.index(obj['label'])
                ply = np.asarray(obj['polygon'], dtype=np.float32)
                left_top = np.min(ply, axis=0)
                right_bottom = np.max(ply, axis=0)

                bbox_list.append([left_top[0], left_top[1], right_bottom[0], right_bottom[1]])
                cls_list.append(cls)
                lbl_list.append(obj['label'])

        if len(bbox_list) <= 0: # no object
            continue
            # cls = 0  # background, no bbox
            # bbox_list.append([0, 0, 0, 0])
            # cls_list.append(cls)    #[0]
            # lbl_list.append(cls_labels[0])

        # if (np.asarray(Image.open(image_path)).ndim) < 3:
        #     print('=======================> {} has one channel'.format(image_path))
        #     continue
        if 'troisdorf_000000_000073_leftImg8bit.png' in image_path:
            print('=======================> {} has one channel'.format(image_path))
            continue

        dataset_list.append({'image_path': image_path,
                             'label': lbl_list,
                             'class': cls_list,
                             'bbox': bbox_list})
    return dataset_list


def load_dataset_list(sample_path, label_path, label_groups):
    # load the each image path
    image_paths = []
    for root, dirs, files in os.walk(sample_path):
        for filename in files:
            image_paths.append(root + '/' + filename)

    # load the each label path
    polygon_paths = []
    for root, dirs, files in os.walk(label_path):
        for filename in files:
            if filename.find('.json') > 0:
                polygon_paths.append(root + '/' + filename)

    # Match two path
    data_paths = []
    for img in image_paths:
        for poly in polygon_paths:
            if img[-22:-16] == poly[-29:-23]:
                data_paths.append((img, poly))
                break

    # extract bounding box
    dataset_list = []
    obj_name = set()
    for image_path, polygon_dir in data_paths:

        if image_path[-22:-16] != polygon_dir[-29:-23]:
            print('-----------?', image_path[-22:-16], polygon_dir[-29:-23])
            continue
        with open(polygon_dir, 'r') as p:
            polygon = json.load(p)

        # gather object names
        obj_name.update([obj['label'] for obj in polygon['objects']])

        # print([obj['label'] for obj in polygon['objects']])
        bbox_list = []
        cls_list = []
        lbl_list = []
        for obj in polygon['objects']:
            for lbl_k in label_groups.keys():
                if obj['label'] in label_groups[lbl_k]:
                    cls = list(label_groups.keys()).index(lbl_k)
                    ply = np.asarray(obj['polygon'], dtype=np.float32)
                    left_top = np.min(ply, axis=0)
                    right_bottom = np.max(ply, axis=0)
                    # print(obj['label'], cls, lbl_k)
                    bbox_list.append([left_top[0], left_top[1], right_bottom[0], right_bottom[1]])
                    cls_list.append(cls)
                    lbl_list.append(lbl_k)

        if len(cls_list) <= 0: # no object
            continue
            # cls = 0  # background, no bbox
            # bbox_list.append([0, 0, 0, 0])
            # cls_list.append(cls)    #[0]
            # lbl_list.append(cls_labels[0])

        # if (np.asarray(Image.open(image_path)).ndim) < 3:
        #     print('=======================> {} has one channel'.format(image_path))
        #     continue
        if 'troisdorf_000000_000073_leftImg8bit.png' in image_path:
            print('=======================> {} has one channel'.format(image_path))
            continue


        # print({'image_path': image_path,
        #                      'label': lbl_list,
        #                      'class': cls_list,
        #                      'bbox': bbox_list})
        dataset_list.append({'image_path': image_path,
                             'label': lbl_list,
                             'class': cls_list,
                             'bbox': bbox_list})

    # print(len(obj_name)) # show total objects numbers
    # print(obj_name) # show all objects
    return dataset_list


def show_loss(data_dir):
    '''
        train_data = {'conf_losses': np.asarray(conf_losses),
                  'loc_losses': np.asarray(loc_losses),
                  'v_conf_losses': np.asarray(v_conf_losses),
                  'v_loc_losses': np.asarray(v_loc_losses),
                  'learning_rate': learning_rate,
                  'total_itr': itr,
                  'max_epoch': max_epoch,
                  'train_time': '%d:%02d:%02d' % (h, m, s)}
    '''
    loss_data = torch.load(data_dir, map_location='cpu')

    plt.plot(loss_data['conf_losses'][:, 0], loss_data['conf_losses'][:, 1], label='train_conf')
    plt.plot(loss_data['loc_losses'][:, 0], loss_data['loc_losses'][:, 1], label='train_regr')
    plt.plot(loss_data['v_conf_losses'][:, 0], loss_data['v_conf_losses'][:, 1], label='valid_conf')
    plt.plot(loss_data['v_loc_losses'][:, 0], loss_data['v_loc_losses'][:, 1], label='valid_regr')
    plt.title('<Training/Validation Loss Curve> \n learning_rate: {}, itr: {} \n max_epoch: {}, train_time: {}'.format(
                loss_data['learning_rate'], loss_data['total_itr'],
                loss_data['max_epoch'], loss_data['train_time']))
    plt.ylabel('MSE loss (0-1)')
    plt.xlabel('number of iterations')
    plt.legend()
    plt.show()
    plt.savefig(data_dir[:-5] + '_loss.png')


def show_log(log_dir):
    log_data = torch.load(log_dir, map_location='cpu')
    print(log_data['log'])

