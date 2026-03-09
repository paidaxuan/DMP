from .vision_transformer import ViT
import torch
import torch.nn as nn

# L2 norm
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_vision_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H,cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)
        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes
        self.class_per_task = [num_classes]

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)


    def forward(self, x, return_feature=False):
        features = self.base(x)
        feat = self.bottleneck(features)
        cls_score = self.classifier(feat)

        if self.training:
            return cls_score, features
        else:
            if return_feature:
                return cls_score, features
            
            return self.l2norm(feat)

    @property
    def feat_dim(self):
        return self.in_planes

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def initial_classifier(self, init_loader, new_classes):

        self.eval()
        features_dict = {}

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(init_loader):
                if len(batch_data) == 2:
                    input, label = batch_data
                elif len(batch_data) == 5:
                    input, _, label, _, _ = batch_data
                elif len(batch_data) == 4:
                    input, _, label, _ = batch_data
                else:
                    raise ValueError(f"Unexpected number of values from dataloader: {len(batch_data)}")
                input = input.cuda()
                feat = self.forward(input)
                if isinstance(feat, tuple):
                    feat = feat[-1]

                for i in range(len(label)):
                    class_id = label[i].item()
                    if class_id not in features_dict:
                        features_dict[class_id] = []
                    features_dict[class_id].append(feat[i].cpu())

        class_centers = []
        for class_id in sorted(features_dict.keys()):
            class_features = torch.stack(features_dict[class_id])
            center = class_features.mean(dim=0)
            class_centers.append(center)

        if len(class_centers) < new_classes:
            print(f"Warning: Only found {len(class_centers)} classes in init_loader, but need {new_classes}")
            while len(class_centers) < new_classes:
                random_center = torch.randn(self.in_planes) * 0.01
                class_centers.append(random_center)

        class_centers = class_centers[:new_classes]
        class_centers = torch.stack(class_centers)

        return class_centers.cuda()

    def add_model(self, new_classes, init_loader):

        old_classes = self.num_classes
        self.num_classes += new_classes
        self.class_per_task.append(new_classes)

        org_classifier_params = self.classifier.weight.data.clone()

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifier.cuda()
        self.classifier.weight.data[:old_classes].copy_(org_classifier_params)

        if init_loader is not None:
            try:
                class_centers = self.initial_classifier(init_loader, new_classes)
                self.classifier.weight.data[old_classes:].copy_(class_centers)
                print(f"Successfully initialized new classes using data from init_loader")
            except Exception as e:
                print(f"Error initializing new classes with init_loader: {e}")
                print("Using random initialization for new classes instead")
        else:
            print("No init_loader provided, using random initialization for new classes")

        self.cuda()

        print(f"Model expanded: {old_classes} -> {self.num_classes} classes")
        print(f"Current class_per_task: {self.class_per_task}")