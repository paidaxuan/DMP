import random

import numpy as np
from PIL import Image
import torch.utils.data as data
import math
import torch

class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform1=None, transform2=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image  = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform1 = transform1
        self.transform2 = transform2
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        if self.cIndex is None and self.tIndex is None:
            img1, target1 = self.train_color_image[index], self.train_color_label[index]
            img2, target2 = self.train_thermal_image[index], self.train_thermal_label[index]
        else:
            img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
            img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform1(img1)
        img2 = self.transform2(img2)

        return img1, img2, target1, target2

    def __len__(self):
        if self.cIndex is not None:
            return len(self.cIndex)
        else:
            return len(self.train_color_label)
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform1=None, transform2=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288),Image.LANCZOS) #Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform1 = transform1
        self.transform2 = transform2
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        if self.cIndex is None and self.tIndex is None:
            img1, target1 = self.train_color_image[index], self.train_color_label[index]
            img2, target2 = self.train_thermal_image[index], self.train_thermal_label[index]
        else:
            img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
            img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform1(img1)
        img2 = self.transform2(img2)

        return img1, img2, target1, target2

    def __len__(self):
        if self.cIndex is not None:
            return len(self.cIndex)
        else:
            return len(self.train_color_label)


class LLCMData(data.Dataset):
    def __init__(self, data_dir, transform1=None, transform2=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_vis.txt'
        train_thermal_list = data_dir + 'idx/train_nir.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 384), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 384), Image.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            # print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform1 = transform1
        self.transform2 = transform2
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        if self.cIndex is None and self.tIndex is None:
            img1, target1 = self.train_color_image[index], self.train_color_label[index]
            img2, target2 = self.train_thermal_image[index], self.train_thermal_label[index]
        else:
            img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
            img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform1(img1)
        img2 = self.transform2(img2)

        return img1, img2, target1, target2

    def __len__(self):
        if self.cIndex is not None:
            return len(self.cIndex)
        else:
            return len(self.train_color_label)
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None,img_size = (144,288),test_cam = None):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        if test_cam is not None:
            self.test_cam = test_cam

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

class ReplayData(data.Dataset):

    def __init__(self,data_source,transforms = None):
        self.dataset = data_source
        self.transform = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        color_img,thermal_img,pid,domain = self.dataset[index]

        if self.transform is not None:
            color_img = self.transform(color_img)
            thermal_img = self.transform(thermal_img)

        return color_img,thermal_img,pid,pid,domain
    
class VideoDataset_train(data.Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir,dataset_rgb, seq_len=12, sample='evenly', transform1=None, transform2=None, index1=[], index2=[], isRandom=True):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform1 = transform1
        self.transform2 = transform2
        self.cIndex = index1
        self.tIndex = index2
        self.isRandom = isRandom

        # Preselect one random image index for each tracklet
        # self.imgIndex_ir = [np.random.choice(len(dataset_ir[i][0])) for i in range(self.dataset_ir)]
        self.imgIndex_ir = [np.random.choice(len(dataset_ir[i][0])) for i in range(len(dataset_ir))]
        self.imgIndex_rgb = [np.random.choice(len(dataset_rgb[i][0])) for i in range(len(dataset_rgb))]

    def __len__(self):
        if self.cIndex is not None:
            return len(self.cIndex)
        else:
            return len(self.dataset_rgb)

    def __getitem__(self, index):
        if not self.isRandom:
            img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.tIndex[index]]
            img_ir_path = img_ir_paths[self.imgIndex_ir[self.tIndex[index]]]

            img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.cIndex[index]]
            img_rgb_path = img_rgb_paths[self.imgIndex_rgb[self.cIndex[index]]]

            img_ir = read_image(img_ir_path)
            img_ir = np.array(img_ir)
            if self.transform2 is not None:
                img_ir = self.transform2(img_ir)

            img_rgb = read_image(img_rgb_path)
            img_rgb = np.array(img_rgb)
            if self.transform1 is not None:
                img_rgb = self.transform1(img_rgb)
            return img_rgb, img_ir, pid_rgb, pid_ir
        else:
            img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.tIndex[index]]

            num_ir = len(img_ir_paths)

            img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.cIndex[index]]
            num_rgb = len(img_rgb_paths)

            S = self.seq_len
            sample_clip_ir = []
            frame_indices_ir = list(range(num_ir))
            if num_ir < S:
                strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (S - num_ir)
                for s in range(S):
                    pool_ir = strip_ir[s * 1:(s + 1) * 1]
                    sample_clip_ir.append(list(pool_ir))
            else:
                inter_val_ir = math.ceil(num_ir / S)
                strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
                for s in range(S):
                    pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                    sample_clip_ir.append(list(pool_ir))

            sample_clip_ir = np.array(sample_clip_ir)

            sample_clip_rgb = []
            frame_indices_rgb = list(range(num_rgb))
            if num_rgb < S:
                strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
                for s in range(S):
                    pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                    sample_clip_rgb.append(list(pool_rgb))
            else:
                inter_val_rgb = math.ceil(num_rgb / S)
                strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
                for s in range(S):
                    pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                    sample_clip_rgb.append(list(pool_rgb))

            sample_clip_rgb = np.array(sample_clip_rgb)

            if self.sample == 'random':
                """
                Randomly sample seq_len consecutive frames from num frames,
                if num is smaller than seq_len, then replicate items.
                This sampling strategy is used in training phase.
                """
                frame_indices = range(num_ir)
                rand_end = max(0, len(frame_indices) - self.seq_len - 1)
                begin_index = random.randint(0, rand_end)
                end_index = min(begin_index + self.seq_len, len(frame_indices))

                indices = frame_indices[begin_index:end_index]
                indices = list(indices)
                for index in indices:
                    if len(indices) >= self.seq_len:
                        break
                    indices.append(index)
                indices = np.array(indices)
                imgs_ir = []
                for index in indices:
                    index = int(index)
                    img_path = img_ir_paths[index]
                    img = read_image(img_path)

                    img = np.array(img)
                    if self.transform2 is not None:
                        img = self.transform2(img)

                    imgs_ir.append(img)
                imgs_ir = torch.cat(imgs_ir, dim=0)

                frame_indices = range(num_rgb)
                rand_end = max(0, len(frame_indices) - self.seq_len - 1)
                begin_index = random.randint(0, rand_end)
                end_index = min(begin_index + self.seq_len, len(frame_indices))

                indices = frame_indices[begin_index:end_index]

                indices = list(indices)
                for index in indices:
                    if len(indices) >= self.seq_len:
                        break

                    indices.append(index)
                indices = np.array(indices)
                imgs_rgb = []
                for index in indices:
                    index = int(index)
                    img_path = img_rgb_paths[index]
                    img = read_image(img_path)

                    img = np.array(img)
                    if self.transform1 is not None:
                        img = self.transform1(img)

                    imgs_rgb.append(img)
                imgs_rgb = torch.cat(imgs_rgb, dim=0)
                return imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb
            elif self.sample == 'video_train':
                idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
                number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]

                imgs_ir = []
                for index in number_ir:
                    index = int(index)
                    img_path = img_ir_paths[index]
                    img = read_image(img_path)

                    img = np.array(img)
                    if self.transform2 is not None:
                        img = self.transform2(img)

                    imgs_ir.append(img)
                imgs_ir = torch.cat(imgs_ir, dim=0)

                idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
                number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
                imgs_rgb = []
                for index in number_rgb:
                    index = int(index)
                    img_path = img_rgb_paths[index]
                    img = read_image(img_path)

                    img = np.array(img)
                    if self.transform1 is not None:
                        img = self.transform1(img)

                    imgs_rgb.append(img)
                imgs_rgb = torch.cat(imgs_rgb, dim=0)
                return imgs_rgb, imgs_ir, pid_rgb, pid_ir
            else:
                raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_test(data.Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None,test_cam = None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.test_cam = test_cam

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)

                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

