import os
import torch
from torch.utils.data import DataLoader
from cityscape_dataset import CityScapeDataset
from ssd_util import load_dataset_list, load_dataset_list_original, show_loss, show_log
from ssd_net import SSD
from ssd_train import train_net
from ssd_test import test_net


if __name__ == '__main__':
    # Define Label Group
    dataset_label_group = {
        'background': [],
        'sm_veh': ['motorcycle', 'motorcyclegroup', 'bicycle', 'bicyclegroup'],
        'med_veh': ['car', 'cargroup'],
        # 'ego_veh': ['ego vehicle'],
        'big_veh': ['bus', 'trailer', 'truck'],
        # 'people': ['person', 'persongroup'],
        # 'riders': ['rider', 'ridergroup']
    }

    # Define Configurations
    config = {'is_gpu': True,
              'debug': False,
              'n_aug': 1,
              'n_batch': 64,
              'n_worker': 4,
              'lr': 0.001,
              'max_epoch': 100,
              'save_epochs': [10,20,30,40,50,60,70,80,90],
              'is_lr_scheduled': False,
              # 'class_labels': ['background', 'cargroup'],
              # 'class_labels': ['background', 'persongroup', 'person', 'cargroup', 'car'],
              'label_groups': dataset_label_group,
              'class_labels': list(dataset_label_group.keys()),
              'is_train': True,
              'is_test': True,
              'results_path': '/home/suhongk/sfuhome/CMPT742/Lab3/vehicle_detection_v2/results/SSD__28th_16:47_best_model.pth'
              }

    # crop original image
    # person + persongroup , car+Cargroup
    # Overfitted data for the unaug
    # check training set

    # Default Cuda Setting -------------------------------------------------
    from torch.multiprocessing import Pool, Process, set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.benchmark = True

    # load dataset_list -------------------------------------------------
    if config['is_gpu']:
        sample_path = '/home/datasets/full_dataset/train_extra/'
        label_path = '/home/datasets/full_dataset_labels/train_extra'
    else:
        sample_path = '../cityscapes_samples/'
        label_path = '../cityscapes_samples_labels/'

    dataset_list = load_dataset_list(sample_path, label_path, config['label_groups'])
    # dataset_list = load_dataset_list_original(sample_path, label_path, config['class_labels'])
    # Define dataset/dataloader -------------------------------------------
    num_train = int(0.3 * len(dataset_list))
    num_valid = int(0.1 * len(dataset_list))
    if config['is_train']:
        train_dataset = CityScapeDataset(dataset_list[:num_train], n_augmented=config['n_aug'], debug=config['debug'])
        train_loader = DataLoader(train_dataset,  batch_size=config['n_batch'], shuffle=True, num_workers=config['n_worker'])
        print('Total training items: ', len(train_dataset))
        print('Total training batches size in one epoch: ', len(train_loader))

        valid_dataset = CityScapeDataset(dataset_list[num_train:(num_train + num_valid)], debug=config['debug'])
        valid_loader = DataLoader(valid_dataset, batch_size=config['n_batch'], shuffle=True, num_workers=config['n_worker'])
        print('Total validating items: ', len(valid_dataset))
        print('Total validating batches size in one epoch: ', len(valid_loader))

    if config['is_test']:
        test_dataset = CityScapeDataset(dataset_list[(num_train + num_valid):], debug=config['debug'])
        print('Total testing items: ', len(test_dataset))

    # Train network -----------------------------------------------------
    if config['is_train']:
        lab_results_dir = "./results/"  # for the results
        results_path = train_net(train_loader, valid_loader, config['class_labels'], lab_results_dir,
                                 learning_rate=config['lr'], is_lr_scheduled=config['is_lr_scheduled'],
                                 max_epoch=config['max_epoch'], save_epochs=config['save_epochs'])
        print('\n\n-----------------------\n\tresult_path:', results_path)
        if not config['is_gpu']:
            show_loss(results_path + '.loss')
            # show_log(results_path + '__train.log')
            # show_log(results_path + '__valid.log')
        if config['is_test']:
            test_net(test_dataset, config['class_labels'], (results_path + '__model.pth'))
    # Train network -----------------------------------------------------
    if config['is_test'] and not config['is_train']:
        test_net(test_dataset, config['class_labels'], config['results_path'])
        # pass
    # Test Code ----------------------------------------------------------
    # idx, (imgs, bbox_label, bbox_indices, _) = next(enumerate(train_loader))
    # print(bbox_indices)
    # test_dataset.__getitem__(9)
    # net = SSD(len(class_labels))
    # net.cuda()
    # net.forward(torch.rand(1, 3, 300, 300))




