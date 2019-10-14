import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from ssd_net import SSD
import torch.nn.functional as F
from bbox_loss import MultiboxLoss
from bbox_helper import loc2bbox, nms_bbox


def test_net(test_dataset, class_labels, results_path):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Load the save model and deploy
    test_net = SSD(len(class_labels))

    test_net_state = torch.load(os.path.join(results_path))
    test_net.load_state_dict(test_net_state)
    test_net.cuda()

    test_net.eval()

    # accuracy
    count_matched = 0
    count_gt = 0
    for test_item_idx in range(0, len(test_dataset)):
        # test_item_idx = random.choice(range(0, len(test_dataset)))
        test_image_tensor, test_label_tensor, test_bbox_tensor, prior_bbox = test_dataset[test_item_idx]

        # run Forward
        with torch.no_grad():
            pred_scores_tensor, pred_bbox_tensor = test_net.forward(test_image_tensor.unsqueeze(0).cuda()) # N C H W

        # scores -> Prob # because I deleted F.softmax~ at the ssd_net for net.eval
        pred_scores_tensor = F.softmax(pred_scores_tensor, dim=2)

        # bbox loc -> bbox (center)
        pred_bbox_tensor = loc2bbox(pred_bbox_tensor, prior_bbox.unsqueeze(0))

        # NMS : return tensor dictionary (bbo
        pred_picked = nms_bbox(pred_bbox_tensor[0], pred_scores_tensor[0]) # not tensor, corner form

        # Show the result
        test_image = test_image_tensor.cpu().numpy().astype(np.float32).transpose().copy()  # H, W, C
        test_image = ((test_image + 1) / 2)
        gt_label = test_label_tensor.cpu().numpy().astype(np.uint8).copy()
        gt_bbox_tensor = torch.cat([test_bbox_tensor[..., :2] - test_bbox_tensor[..., 2:] / 2,
                                    test_bbox_tensor[..., :2] + test_bbox_tensor[..., 2:] / 2], dim=-1)
        gt_bbox = gt_bbox_tensor.detach().cpu().numpy().astype(np.float32).reshape((-1, 4)).copy() * 300
        gt_idx = gt_label > 0

        # Calculate accuracy
        pred_scores = pred_scores_tensor.detach().cpu().numpy().astype(np.float32).copy()
        pred_label = pred_scores[0].argmax(axis=1)

        n_matched = 0
        for gt, pr in zip(gt_label, pred_label):
            if gt > 0 and gt == pr:
                n_matched += 1
        acc_per_image = 100 * n_matched / gt_idx.sum()
        count_matched += n_matched
        count_gt += gt_idx.sum()



        # Show the results
        gt_bbox = gt_bbox[gt_idx]
        gt_label = gt_label[gt_idx]
        if True:
            for idx in range(gt_bbox.shape[0]):
                cv2.rectangle(test_image, (gt_bbox[idx][0], gt_bbox[idx][1]), (gt_bbox[idx][2], gt_bbox[idx][3]), (255, 0, 0), 1)
                cv2.putText(test_image, str(gt_label[idx]), (gt_bbox[idx][0], gt_bbox[idx][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1, cv2.LINE_AA)

                #--------------------
                # cv2.rectangle(test_image, (pred_bbox[idx][0], pred_bbox[idx][1]), (pred_bbox[idx][2], pred_bbox[idx][3]),
                #               (0, 255, 0), 1)

                #-----------------------

            for cls_dict in pred_picked:
                for p_score, p_bbox in zip(cls_dict['picked_scores'], cls_dict['picked_bboxes']):
                    p_lbl = '%d | %.2f' % (cls_dict['class'], p_score)
                    p_bbox = p_bbox * 300
                    print(p_bbox, p_lbl)
                    cv2.rectangle(test_image, (p_bbox[0], p_bbox[1]), (p_bbox[2], p_bbox[3]),
                                  (0, 0, 255), 2)
                    cv2.putText(test_image,  p_lbl, (p_bbox[0], p_bbox[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                            1, cv2.LINE_AA)
            plt.imshow(test_image)
            plt.suptitle(class_labels)
            plt.title('Temp Accuracy: {} %'.format(acc_per_image))
            plt.show()

    acc = 100 * count_matched / count_gt
    print('Classification acc: ', acc, '%')

