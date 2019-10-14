import numpy as np
from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt
import torch.nn
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors, iou


class CityScapeDataset(Dataset):

    def __init__(self, dataset_list, n_augmented=0, debug=False):
        self.dataset_list = dataset_list

        # TODO: implement prior bounding box
        prior_layer_cfg = [
            {'layer_name': 'Conv6', 'feature_dim_hw': (19, 19), 'bbox_size': (30, 30),
             'aspect_ratio': (1.0, 1 / 2, 1 / 4, 2.0, 4.0, np.sqrt(1.16))},
            {'layer_name': 'Conv12', 'feature_dim_hw': (10, 10), 'bbox_size': (78, 78),
             'aspect_ratio': (1.0, 1 / 2, 1 / 4, 2.0, 4.0, np.sqrt(1.16))},
            {'layer_name': 'Conv8_2', 'feature_dim_hw': (5, 5), 'bbox_size': (126, 126),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, np.sqrt(1.16))},
            {'layer_name': 'Conv9_2', 'feature_dim_hw': (3, 3), 'bbox_size': (174, 174),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, np.sqrt(1.16))},
            {'layer_name': 'Conv10_2', 'feature_dim_hw': (2, 2), 'bbox_size': (222, 222),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, np.sqrt(1.16))},
            {'layer_name': 'Conv11_2', 'feature_dim_hw': (1, 1), 'bbox_size': (270, 270),
             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, np.sqrt(1.16))}
        ]

        self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg)

        # Pre-process parameters:
        #  Normalize: (I-self.mean)/self.std
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0
        self.n_augmented = n_augmented
        self.net_size = (300, 300)
        self.debug = debug

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list) * (self.n_augmented if self.n_augmented > 0 else 1)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4)
        :return bbox_label: matched classification label, dim: (num_priors)
        """
        sample_idx = (lambda i, n: i // n if n is not 0 else i)(idx, self.n_augmented)
        sample = self.dataset_list[sample_idx]

        # TODO: implement data loading
        # 1. Load image as well as the bounding box with its label
        image = Image.open(sample['image_path'])
        label = sample['label']
        cls = sample['class']
        bbox = sample['bbox']

        # 2. convert the image and bbox to numpy array and crop to square form
        image = np.asarray(image, dtype=np.uint8)
        bbox_cr = np.asarray(bbox, dtype=np.float32)
        show_list = [{'image': image.copy(), 'bbox_cr': bbox_cr.copy(), 'label': label.copy(), 'title': 'Original'}]
        # image, bbox_cr, cls, label = self.crop(image, bbox_cr, cls, label, is_random=False)
        # show_list.append({'image': image.copy(), 'bbox_cr': bbox_cr.copy(), 'label': label.copy(), 'title': 'Cropping(Square)'})

        # 3. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box
        if self.n_augmented > 0:

            # calling the cropping function
            image, bbox_cr, cls, label = self.crop(image, bbox_cr, cls, label, is_random=True)
            show_list.append({'image': image.copy(), 'bbox_cr': bbox_cr.copy(), 'label': label.copy(), 'title': 'Cropping(Random)'})

            # calling the flip function
            image, bbox_cr = self.flip(image, bbox_cr)
            show_list.append(
                {'image': image.copy(), 'bbox_cr': bbox_cr.copy(), 'label': label.copy(), 'title': 'Flipping'})

        # 4. resize the image (H, W, C) to net size(300, 300)
        bbox_cr[:, [0, 2]] = bbox_cr[:, [0, 2]] * (self.net_size[0] / image.shape[1])   # Width
        bbox_cr[:, [1, 3]] = bbox_cr[:, [1, 3]] * (self.net_size[1] / image.shape[0])   # Height
        image = cv2.resize(image, dsize=self.net_size, interpolation=cv2.INTER_CUBIC)
        show_list.append({'image': image.copy(), 'bbox_cr': bbox_cr.copy(), 'label': label.copy(), 'title': 'Resizing'})

        # Check intermediate input
        if self.debug:
            self.show_image(show_list)

        # 5. Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        center_xy = (bbox_cr[:, 2:] + bbox_cr[:, :2]) / 2.
        center_wh = (bbox_cr[:, 2:] - bbox_cr[:, :2])
        bbox_ct = np.concatenate((center_xy, center_wh), axis=1)

        # 6. Normalize the image with self.mean and self.std
        image_norm = np.divide((np.asarray(image, dtype=np.float32) - self.mean), self.std)
        # Normalize the bounding box position value from 0 to 1
        bbox_ct[:, [0, 2]] = bbox_ct[:, [0, 2]] / self.net_size[0]
        bbox_ct[:, [1, 3]] = bbox_ct[:, [1, 3]] / self.net_size[1]

        # 7. Do the matching prior and generate ground-truth labels as well as the boxes
        sample_labels = torch.from_numpy(np.asarray(cls)).type(torch.long)  # Cuda Tensor
        sample_bboxes = torch.from_numpy(bbox_ct).type(torch.float32)  # Cuda Tensor
        if torch.cuda.is_available():
            sample_labels = sample_labels.cuda()
            sample_bboxes = sample_bboxes.cuda()

        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, sample_bboxes, sample_labels, iou_threshold=0.5)
        sample_img_tensor = torch.from_numpy(image_norm.transpose()).type(torch.float32)  # Cuda Tensor

        if torch.cuda.is_available():
            bbox_tensor = bbox_tensor.cuda()
            bbox_label_tensor = bbox_label_tensor.cuda()
            sample_img_tensor = sample_img_tensor.cuda()

        # Check the final tensor input
        if self.debug:
            self.show_tensor_image(sample_img_tensor.clone(),
                                   bbox_tensor.clone(),
                                   bbox_label_tensor.clone(),
                                   label.copy(), sample_idx)

        # [DEBUG] check the output.
        assert isinstance(sample_img_tensor, torch.Tensor)
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        return sample_img_tensor, bbox_label_tensor, bbox_tensor, self.prior_bboxes

    # cropping the image
    @staticmethod
    def crop(image, boxes, cls, labels, is_random=False):
        image_temp = image.copy()
        boxes_temp = boxes.copy()
        cls_temp = cls.copy()
        labels_temp = labels.copy()

        height, width, _ = image.shape

        if len(boxes) == 0:
            # print('No object')
            return image, boxes

        while True:
            if is_random:
                mode = random.choice((
                    None,
                    (0.1, None),
                    (0.3, None),
                    (0.7, None),
                    (0.9, None),
                    (None, None),
                ))
            else:
                mode = (None, None)

            if mode is None:
                # print('No cropping is needed')
                return image, boxes, cls, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            for _ in range(50):
                if is_random:
                    width_crop = random.randrange(int(0.3 * width), width)
                    height_crop = random.randrange(int(0.3 * height), height)
                else: # crop input as rectangle
                    width_crop = np.minimum(width, height)
                    height_crop = np.minimum(width, height)

                # check to see the image has reasonable size (not too narrow vertically and horizontally)
                if height_crop / width_crop < 0.5 or 2 < height_crop / width_crop:
                    continue

                left_top = 0 if width == width_crop else random.randrange(width - width_crop)
                right_bottom = 0 if height == height_crop else random.randrange(height - height_crop)
                roi = np.array((left_top, right_bottom, left_top + width_crop, right_bottom + height_crop))
                # new image corners
                image_crop = image[roi[1]:roi[3], roi[0]:roi[2]]
                # calling the iou function to check if there is any overlap between our ROI and ground truth boxes
                iou_output = iou(torch.from_numpy(boxes.copy()).float(), torch.from_numpy(roi[np.newaxis]).float())
                # check if iou_output is larger than the treshhold
                if not (min_iou <= iou_output.min() and iou_output.max() <= max_iou):
                    continue

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                    .all(axis=1)
                boxes = boxes[mask].copy()
                labels = [labels[i] for i in range(len(mask)) if mask[i]]
                cls = [cls[i] for i in range(len(mask)) if mask[i]]
                width2, height2, _ = image_crop.shape
                # adjusting the boxesz to the new image
                boxes[:, :2] = np.maximum(boxes[:, :2], roi[:2])
                boxes[:, :2] -= roi[:2]
                boxes[:, 2:] = np.minimum(boxes[:, 2:], roi[2:])
                boxes[:, 2:] -= roi[:2]

                if len(cls) < 1: # if cls is 0, recover every input to original
                    image_crop = image_temp
                    boxes = boxes_temp
                    cls = cls_temp
                    labels = labels_temp
                return image_crop, boxes, cls, labels

    # flipping the image
    @staticmethod
    def flip(image, boxes):
        _, width, _ = image.shape
        if random.randrange(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, boxes

    @staticmethod
    def show_image(show_list):
        fig = plt.figure()
        for i, item in zip(range(len(show_list)), show_list):
            image = item['image']
            label = item['label']
            bbox_cr = item['bbox_cr']
            ax = plt.subplot(1, len(show_list), i + 1)
            for idx in range(0, len(label)):
                cv2.rectangle(image, (bbox_cr[idx][0], bbox_cr[idx][1]),
                              (bbox_cr[idx][2], bbox_cr[idx][3]), (255, 0, 0), 2)
                cv2.putText(image, label[idx], (bbox_cr[idx][0], bbox_cr[idx][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1, cv2.LINE_AA)
            ax.imshow(image)
            ax.set_title(item['title'])
        fig.tight_layout()
        plt.show()

    def show_tensor_image(self, img_tensor, bbox_tensor, bbox_label_tensor, label, sample_idx):
        image_s = img_tensor[:, :, :].cpu().numpy().astype(np.float32).transpose().copy()  # c , h, w -> h, w, c
        image_s = ((image_s + 1) / 2)
        bbox_cr_s = torch.cat([bbox_tensor[..., :2] - bbox_tensor[..., 2:] / 2,
                               bbox_tensor[..., :2] + bbox_tensor[..., 2:] / 2], dim=-1)
        bbox_prior_s = bbox_cr_s[:].cpu().numpy().astype(np.float32).reshape((-1, 4)).copy()  # First sample in batch
        bbox_prior_s = (bbox_prior_s * 300)
        label_prior_s = bbox_label_tensor[:].cpu().numpy().astype(np.float32).copy()
        bbox_s = bbox_prior_s[label_prior_s > 0]
        label_s = (label_prior_s[label_prior_s > 0]).astype(np.uint8)

        print("Total matched prior objects: ", len(label_s))
        print(self.dataset_list[sample_idx]['image_path'])
        print("Sample Index", sample_idx)
        print(label)
        for idx in range(0, len(label_s)):
            cv2.rectangle(image_s, (bbox_s[idx][0], bbox_s[idx][1]), (bbox_s[idx][2], bbox_s[idx][3]), (255, 0, 0), 2)
            cv2.putText(image_s, str(label_s[idx]), (bbox_s[idx][0], bbox_s[idx][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 0, 0), 1, cv2.LINE_AA)
        plt.imshow(image_s)
        plt.title('Tensor Image Input')
        plt.show()