import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
from ssd_net import SSD
import torch.nn.functional as F
from bbox_helper import loc2bbox, nms_bbox
from cityscape_dataset import CityScapeDataset

if __name__ == '__main__':
    # Choose model
    results_path = './results/SSD__29th_02:01_best_model.pth'

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

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # 1. Conver image to input image tensor -------------------------------------------
    img_file_path = sys.argv[1]
    #  # the index should be 1, 0 is the 'eval.py'
    image = Image.open(img_file_path).resize((300, 300))
    image = np.divide((np.asarray(image, dtype=np.float32) - 128.0), np.asarray((127, 127, 127)))
    img_tensor = torch.from_numpy(image.transpose()).type(torch.float32)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # 2. Load the saved model and test ------------------------------------------------
    class_labels = list(dataset_label_group.keys())
    test_net = SSD(len(class_labels))

    test_net_state = torch.load(os.path.join(results_path))
    test_net.load_state_dict(test_net_state)

    if torch.cuda.is_available():
        test_net.cuda()

    test_net.eval()

    # 3. Run Forward -------------------------------------------------------------------
    with torch.no_grad():
        pred_scores_tensor, pred_bbox_tensor = test_net.forward(img_tensor.unsqueeze(0))  # N C H W

    prior = CityScapeDataset([])
    prior_bbox = prior.get_prior_bbox()

    pred_scores_tensor = F.softmax(pred_scores_tensor, dim=2)   # eval mode softmax was disabled in ssd_test
    pred_bbox_tensor = loc2bbox(pred_bbox_tensor, CityScapeDataset([]).get_prior_bbox().unsqueeze(0))
    pred_picked = nms_bbox(pred_bbox_tensor[0], pred_scores_tensor[0])

    # 4. plot result
    test_image = img_tensor.cpu().numpy().astype(np.float32).transpose().copy()  # H, W, C
    test_image = ((test_image + 1) / 2)

    for cls_dict in pred_picked:
        for p_score, p_bbox in zip(cls_dict['picked_scores'], cls_dict['picked_bboxes']):
            p_lbl = '%s | %.2f' % (class_labels[cls_dict['class']], p_score)
            p_bbox = p_bbox * 300
            print(p_bbox, p_lbl)
            cv2.rectangle(test_image, (p_bbox[0], p_bbox[1]), (p_bbox[2], p_bbox[3]),
                          (0, 0, 255), 2)
            cv2.putText(test_image, p_lbl, (p_bbox[0], p_bbox[1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150),
                        1, cv2.LINE_AA)
    plt.imshow(test_image)
    plt.suptitle(class_labels)
    # plt.title(acc)
    plt.show()
