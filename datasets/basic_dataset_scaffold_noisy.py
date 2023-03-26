from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from numpy.testing import assert_array_almost_equal


"""==================================================================================================="""
################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseDataset(Dataset):
    def __init__(self, image_dict, opt, is_validation=False):
        self.is_validation = is_validation
        self.pars          = opt

        #####
        self.image_dict = image_dict

        #####
        self.init_setup(opt)


        #####
        if 'bninception' not in opt.arch:
            self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        else:
            # normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[1., 1., 1.])
            self.f_norm = normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[0.0039, 0.0039, 0.0039])

        transf_list = []

        self.crop_size = crop_im_size = 224 if 'googlenet' not in opt.arch else 227
        if opt.augmentation=='big':
            crop_im_size = 256

        #############
        self.normal_transform = []
        if not self.is_validation:
            if opt.augmentation=='base' or opt.augmentation=='big':
                self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomHorizontalFlip(0.5)])
            elif opt.augmentation=='adv':
                self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomGrayscale(p=0.2),
                                              transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), transforms.RandomHorizontalFlip(0.5)])
            elif opt.augmentation=='red':
                self.normal_transform.extend([transforms.Resize(size=256), transforms.RandomCrop(crop_im_size), transforms.RandomHorizontalFlip(0.5)])
        else:
            self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(crop_im_size)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)


    def init_setup(self, opt):
        self.n_files       = np.sum([len(self.image_dict[key]) for key in self.image_dict.keys()])
        self.avail_classes = sorted(list(self.image_dict.keys()))


        counter = 0
        temp_image_dict = {}
        for i,key in enumerate(self.avail_classes):
            temp_image_dict[key] = []
            for path in self.image_dict[key]:
                temp_image_dict[key].append([path, counter])
                counter += 1

        self.image_dict = temp_image_dict
        self.image_list = [[(x[0],key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        if opt.noisy_label:
            self.image_list = self.noisy_prepare(self.image_list, len(self.avail_classes), opt.noisy_ratio)

            counter = 0
            temp_image_dict = {}

            for i, key in enumerate(self.avail_classes):
                temp_image_dict[key] = []
            for (path, key) in self.image_list:
                    temp_image_dict[key].append([path, counter])
                    counter += 1

            self.image_dict = temp_image_dict

        self.image_paths = self.image_list

        self.is_init = True


    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    ########### symm noisy #######################
    def noisy_prepare(self, image_list, class_num, noisy_ratio):
        label_list = [x[-1] for x in image_list]
        path_list = [x[0] for x in image_list]

        # symmflip
        # P = np.eye(class_num)
        # n = noisy_ratio

        # pairflip
        # P[0, 0], P[0, 1] = 1. - n, n
        # for i in range(1, class_num - 1):
        #     P[i, i], P[i, i + 1] = 1. - n, n
        # P[class_num - 1, class_num - 1], P[class_num - 1, 0] = 1. - n, n

        print('Sym noisy!')

        P = np.ones((class_num, class_num))
        n = noisy_ratio
        P = (n / (class_num - 1)) * P

        P[0, 0] = 1. - n
        for i in range(1, class_num - 1):
            P[i, i] = 1. - n
        P[class_num - 1, class_num - 1] = 1. - n

        new_label_list = self.symm_noisy(np.array(label_list), P, self.pars.seed)
        new_label_list = new_label_list.tolist()

        new_list = []

        for i in range(len(label_list)):
            new_list.append((path_list[i], new_label_list[i]))

        return new_list

    def symm_noisy(self, y, P, random_state = 0):
        """ Flip classes according to transition probability matrix T.
            It expects a number between 0 and the number of classes - 1.
            y: label list
            P: probability matrix
            random_state: random seed
            """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    ################################################


    def __getitem__(self, idx):
        input_image = self.ensure_3dim(Image.open(self.image_list[idx][0]))

        ### Basic preprocessing.
        im_a = self.normal_transform(input_image)
        if 'bninception' in self.pars.arch:
            im_a = im_a[range(3)[::-1],:]
        return self.image_list[idx][-1], im_a, idx


    def __len__(self):
        return self.n_files
