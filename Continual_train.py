from __future__ import absolute_import,print_function
import argparse
import copy

import torch.nn
from torch.backends import cudnn
from utils import *
from data_loader import *
from data_manager import *
from trainer import Trainer
from eval_metrics import eval_func2
from tensorboardX import SummaryWriter
from config.config import cfg

from reid.models.make_model import build_vision_transformer
from reid.utils.lr_scheduler import create_scheduler
from reid.utils.make_optimizer import make_optimizer

from transforms import transform_rgb, transform_rgb2gray, transform_thermal, transform_test
from collections import defaultdict
from prettytable import PrettyTable

def extract_dual_prototypes(model, data_loader, add_num=0, device='cuda'):
    model.eval()

    features_buffer = defaultdict(lambda: {'rgb': [], 'ir': []})

    print("==> Extracting Dual-Modality Prototypes...")
    with torch.no_grad():
        for batch_idx, (input1, input2, label1, label2) in enumerate(data_loader):
            input1, input2 = input1.to(device), input2.to(device)

            if len(input1.size()) == 5:
                input1 = input1.squeeze(1)
                input2 = input2.squeeze(1)

            _, feat1 = model(input1, return_feature=True)  # RGB features
            _, feat2 = model(input2, return_feature=True)  # IR features

            for i, pid in enumerate(label1):
                global_pid = pid.item() + add_num
                features_buffer[global_pid]['rgb'].append(feat1[i].cpu().numpy())

            for i, pid in enumerate(label2):
                global_pid = pid.item() + add_num
                features_buffer[global_pid]['ir'].append(feat2[i].cpu().numpy())

    prototypes = {}
    
    for pid, feats in features_buffer.items():
        if len(feats['rgb']) == 0 or len(feats['ir']) == 0:
            continue

        mean_rgb = np.mean(np.vstack(feats['rgb']), axis=0)
        mean_ir = np.mean(np.vstack(feats['ir']), axis=0)

        prototypes[pid] = {
            'rgb': torch.from_numpy(mean_rgb).float().to(device),
            'ir': torch.from_numpy(mean_ir).float().to(device)
        }

    print(f"==> Extracted prototypes for {len(prototypes)} classes.")
    return prototypes


def main():
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)                   
        np.random.seed(seed)                 
        torch.manual_seed(seed)              
        torch.cuda.manual_seed(seed)         
        torch.cuda.manual_seed_all(seed)     
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    main_worker()

def main_worker():

    print("==========\nArgs:{}\n==========".format(args))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # modify this suffix according to yours
    suffix = f""
    log_name = "log_" + suffix + ".txt"
    sys.stdout = Logger(osp.join(args.logs_dir, log_name))

    save_model_dir = osp.join(args.save_model,suffix)
    if not osp.isdir(save_model_dir + '/'):
        os.makedirs(save_model_dir+'/')

    vis_log_dir = osp.join(args.vis_logs_dir,suffix, "stag1/")
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    writer = SummaryWriter(vis_log_dir)

    # read parameters
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    #change the training order by modifying the order of datasets below
    datasets = ["RegDB","SYSU-MM01","LLCM","VCM"]

    # dual_prototypes
    all_prototypes = {}

    #Create data and dataloaders
    dataset_regdb,num_classes_regdb,gallery_loader_regdb,\
        query_loader_regdb,init_loader_regdb = get_data(datasets[0],args,transform_rgb,transform_thermal,transform_test)

    root = osp.join(args.data_dir,"RegDB/")
    trainset_gray = RegDBData(root, args.trial, transform1=transform_rgb2gray, transform2=transform_thermal)
    color_pos_gray, thermal_pos_gray = GenIdx(trainset_gray.train_color_label, trainset_gray.train_thermal_label)

    ngallery = len(gallery_loader_regdb.dataset.test_label)
    nquery = len(query_loader_regdb.dataset.test_label)

    print('Dataset {} statistics:'.format(datasets[0]))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(num_classes_regdb, len(dataset_regdb.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(num_classes_regdb, len(dataset_regdb.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(unique(query_loader_regdb.dataset.test_label)),nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(unique(gallery_loader_regdb.dataset.test_label)),ngallery))
    print('  ------------------------------')

    #Create model
    start_epoch = 0
    model = build_vision_transformer(num_classes_regdb, cfg)

    if args.step == 1 and args.resume != "":
        model_path = osp.join(save_model_dir,args.resume)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    model.to(device)

    #train settings
    names = ["regdb"]
    test_loaders = [(gallery_loader_regdb,query_loader_regdb)]

    # initialize Opitimizer and lr
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    #store the results of every stage
    all_cmc_perstage = collections.defaultdict(list)
    all_mAP_perstage = collections.defaultdict(list)

    # Start training
    print('Continual training starts!')
    trainer = Trainer(model, cfg)
    color_pos, thermal_pos = GenIdx(dataset_regdb.train_color_label, dataset_regdb.train_thermal_label)

    for epoch in range(start_epoch, 15):
        current_lr = optimizer.param_groups[0]["lr"]

        sampler = IdentitySampler(dataset_regdb.train_color_label, dataset_regdb.train_thermal_label, color_pos,
                                  thermal_pos,
                                  num_pos=args.num_pos, batchSize=args.batch_size)
        dataset_regdb.cIndex = sampler.index1
        dataset_regdb.tIndex = sampler.index2

        if cfg.METHOD == 'PMT':
            if epoch <= cfg.PL_EPOCH:
                sampler_gray = IdentitySampler(trainset_gray.train_color_label, trainset_gray.train_thermal_label,
                                            color_pos_gray, thermal_pos_gray, num_pos=args.num_pos, batchSize=args.batch_size)  # Gray
                # Gray-IR
                trainset_gray.cIndex = sampler_gray.index1
                trainset_gray.tIndex = sampler_gray.index2
                trainloader = data.DataLoader(trainset_gray, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler_gray, drop_last=True)

            else:
                trainloader = data.DataLoader(dataset_regdb, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        else:
            trainloader = data.DataLoader(dataset_regdb, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        trainer.train(epoch, trainloader, optimizer, 1, 0, writer=writer)

        scheduler.step(epoch)

        if (epoch + 1) % 2 == 0:
            print(f"==> Test at epoch {epoch}")
            all_cmc = []
            all_mAP = []

            table = PrettyTable(["Dataset", "Epoch", "Rank-1", "mAP"])
            table.float_format = ".2"

            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP_regdb = eval_func2(model, name, test_loader, model.feat_dim, 1, epoch)
                all_cmc.append(cmc)
                all_mAP.append(mAP_regdb)

                table.add_row([name, epoch, f"{cmc[0]:.2%}", f"{mAP_regdb:.2%}"])

            print(table)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': all_mAP,
            }, True, fpath=osp.join(save_model_dir, 'working_checkpoint_step_1.pth.tar'))

            all_cmc_perstage["stag1"].append(all_cmc)
            all_mAP_perstage["stag1"].append(all_mAP)

        writer.add_scalar("lr", current_lr, epoch)

    stage1_protos = extract_dual_prototypes(model, trainloader, add_num=0, device=device)
    all_prototypes.update(stage1_protos)

    del trainloader, dataset_regdb, optimizer, trainer

    # start to train next dataset
    dataset_sysu,num_classes_sysu,gallery_loader_sysu,\
        query_loader_sysu,init_loader_sysu = get_data(datasets[1],args,transform_rgb,transform_thermal,transform_test)

    root = osp.join(args.data_dir,"SYSU-MM01/")
    trainset_gray = SYSUData(root, transform1=transform_rgb2gray, transform2=transform_thermal)
    color_pos_gray, thermal_pos_gray = GenIdx(trainset_gray.train_color_label, trainset_gray.train_thermal_label)

    ngallery = len(gallery_loader_sysu.dataset.test_label)
    nquery = len(query_loader_sysu.dataset.test_label)

    print('Dataset {} statistics:'.format(datasets[1]))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(num_classes_sysu, len(dataset_sysu.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(num_classes_sysu, len(dataset_sysu.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(unique(query_loader_sysu.dataset.test_label)),nquery))
    print('  query    | {:5d} | {:8d}'.format(len(unique(gallery_loader_sysu.dataset.test_label)),ngallery))
    print('  ------------------------------')

    if args.step == 2 and args.resume != "":
        model_path = osp.join(save_model_dir,args.resume)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # add expandable classifier
    model.add_model(num_classes_sysu,init_loader_sysu)
    add_num = sum(model.class_per_task[:-1])

    # Create old frozen model
    old_model = copy.deepcopy(model)
    old_model = old_model.cuda()
    old_model.eval()

    #train settings
    names.append("sysu")
    test_loaders.append((gallery_loader_sysu, query_loader_sysu))

    # initialize Opitimizer and lr
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    vis_log_dir = osp.join(args.vis_logs_dir,suffix, "stag2/")
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    writer = SummaryWriter(vis_log_dir)

    trainer = Trainer(model, cfg)
    trainer.update_prototypes(all_prototypes)
    color_pos, thermal_pos = GenIdx(dataset_sysu.train_color_label, dataset_sysu.train_thermal_label)

    for epoch in range(start_epoch, 20):
        current_lr = optimizer.param_groups[0]["lr"]

        sampler = IdentitySampler(dataset_sysu.train_color_label, dataset_sysu.train_thermal_label, color_pos,
                                  thermal_pos,
                                  num_pos=args.num_pos, batchSize=args.batch_size)
        dataset_sysu.cIndex = sampler.index1
        dataset_sysu.tIndex = sampler.index2

        if cfg.METHOD == 'PMT':
            if epoch <= cfg.PL_EPOCH:
                sampler_gray = IdentitySampler(trainset_gray.train_color_label, trainset_gray.train_thermal_label,
                                            color_pos_gray, thermal_pos_gray, num_pos=args.num_pos, batchSize=args.batch_size)  # Gray
                # Gray-IR
                trainset_gray.cIndex = sampler_gray.index1
                trainset_gray.tIndex = sampler_gray.index2
                trainloader = data.DataLoader(trainset_gray, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler_gray, drop_last=True)

            else:
                trainloader = data.DataLoader(dataset_sysu, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        else:
            trainloader = data.DataLoader(dataset_sysu, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        trainer.train(epoch, trainloader, optimizer, training_phase=2,
                      add_num=add_num, old_model=old_model, replay=False,writer = writer)

        scheduler.step(epoch)

        if (epoch + 1) % 2 == 0:
            print(f"==> Evaluating Stage 2 at epoch {epoch}...")
            current_epoch_cmc_full = []
            current_epoch_r1 = []
            current_epoch_mAP = []

            table = PrettyTable(["Stage 2 Dataset", "Rank-1", "mAP"])

            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP = eval_func2(model, name, test_loader, model.feat_dim, 2, epoch)
                
                current_epoch_cmc_full.append(cmc)
                current_epoch_r1.append(cmc[0])
                current_epoch_mAP.append(mAP)

                table.add_row([name, f"{cmc[0]:.2%}", f"{mAP:.2%}"])

            avg_r1 = sum(current_epoch_r1) / len(current_epoch_r1)
            avg_mAP = sum(current_epoch_mAP) / len(current_epoch_mAP)

            table.add_row(["-" * 10, "-" * 10, "-" * 10])
            table.add_row(["AVERAGE", f"{avg_r1:.2%}", f"{avg_mAP:.2%}"])

            print(table)

            all_cmc_perstage["stag2"].append(current_epoch_cmc_full)
            all_mAP_perstage["stag2"].append(current_epoch_mAP)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': avg_mAP,
                'rank1': avg_r1,
            }, True, fpath=osp.join(save_model_dir, 'working_checkpoint_step_2.pth.tar'))

        writer.add_scalar("lr", current_lr, epoch)

    stage2_protos = extract_dual_prototypes(model, trainloader, add_num=add_num, device=device)
    all_prototypes.update(stage2_protos)

    del trainloader, dataset_sysu, optimizer, init_loader_sysu, trainer, old_model

    # start to train next dataset
    dataset_llcm,num_classes_llcm,gallery_loader_llcm,\
        query_loader_llcm,init_loader_llcm= get_data("LLCM",args,transform_rgb,transform_thermal,transform_test)

    root = osp.join(args.data_dir,"LLCM/")
    trainset_gray = LLCMData(root, transform1=transform_rgb2gray, transform2=transform_thermal)
    color_pos_gray, thermal_pos_gray = GenIdx(trainset_gray.train_color_label, trainset_gray.train_thermal_label)

    nquery = len(query_loader_llcm.dataset.test_label)
    ngallery = len(gallery_loader_llcm.dataset.test_label)

    print('Dataset {} statistics:'.format(datasets[1]))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(num_classes_llcm, len(dataset_llcm.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(num_classes_llcm, len(dataset_llcm.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(unique(query_loader_llcm.dataset.test_label)),nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(unique(gallery_loader_llcm.dataset.test_label)),ngallery))
    print('  ------------------------------')
    
    if args.step == 3 and args.resume !="":
        model_path = osp.join(save_model_dir, args.resume)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # add expandable classifier
    model.add_model(num_classes_llcm,init_loader_llcm)
    add_num = sum(model.class_per_task[:-1])

    # Create old frozen model
    old_model = copy.deepcopy(model)
    old_model = old_model.cuda()
    old_model.eval()

    names.append("llcm")
    test_loaders.append((gallery_loader_llcm, query_loader_llcm))

    # Re-initialize the optimizer
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    vis_log_dir = osp.join(args.vis_logs_dir,suffix,"stag3/")
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    writer = SummaryWriter(vis_log_dir)

    trainer = Trainer(model, cfg)
    trainer.update_prototypes(all_prototypes)
    color_pos, thermal_pos = GenIdx(dataset_llcm.train_color_label, dataset_llcm.train_thermal_label)

    for epoch in range(start_epoch, 20):
        current_lr = optimizer.param_groups[0]["lr"]

        sampler = IdentitySampler(dataset_llcm.train_color_label, dataset_llcm.train_thermal_label, color_pos,
                                thermal_pos, num_pos=args.num_pos, batchSize=args.batch_size)
        dataset_llcm.cIndex = sampler.index1
        dataset_llcm.tIndex = sampler.index2

        if cfg.METHOD == 'PMT':
            if epoch <= cfg.PL_EPOCH:
                sampler_gray = IdentitySampler(trainset_gray.train_color_label, trainset_gray.train_thermal_label,
                                            color_pos_gray, thermal_pos_gray, num_pos=args.num_pos, batchSize=args.batch_size)  # Gray
                # Gray-IR
                trainset_gray.cIndex = sampler_gray.index1
                trainset_gray.tIndex = sampler_gray.index2
                trainloader = data.DataLoader(trainset_gray, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler_gray, drop_last=True)

            else:
                trainloader = data.DataLoader(dataset_llcm, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        else:
            trainloader = data.DataLoader(dataset_llcm, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        trainer.train(epoch, trainloader, optimizer, training_phase=3,
                      add_num=add_num, old_model=old_model, replay=False,writer = writer)

        scheduler.step(epoch)

        if (epoch + 1) % 2 == 0:
            print(f"==> Evaluating Stage 3 at epoch {epoch}...")

            current_epoch_cmc_full = []
            current_epoch_r1 = []
            current_epoch_mAP = []

            table = PrettyTable(["Stage 3 Dataset", "Rank-1", "mAP"])

            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP = eval_func2(model, name, test_loader, model.feat_dim, 3, epoch)

                current_epoch_cmc_full.append(cmc)
                current_epoch_r1.append(cmc[0])
                current_epoch_mAP.append(mAP)

                table.add_row([name, f"{cmc[0]:.2%}", f"{mAP:.2%}"])

            avg_r1 = sum(current_epoch_r1) / len(current_epoch_r1)
            avg_mAP = sum(current_epoch_mAP) / len(current_epoch_mAP)

            table.add_row(["-" * 10, "-" * 10, "-" * 10])
            table.add_row(["AVERAGE", f"{avg_r1:.2%}", f"{avg_mAP:.2%}"])

            print(table)

            all_cmc_perstage["stag3"].append(current_epoch_cmc_full)
            all_mAP_perstage["stag3"].append(current_epoch_mAP)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': avg_mAP,
                'rank1': avg_r1,
            }, True, fpath=osp.join(save_model_dir, 'working_checkpoint_step_3.pth.tar'))

        writer.add_scalar("lr", current_lr, epoch)

    stage3_protos = extract_dual_prototypes(model, trainloader, add_num=add_num, device=device)
    all_prototypes.update(stage3_protos)

    del trainloader, trainer, optimizer, init_loader_llcm,dataset_llcm,old_model

    dataset_vcm,num_classes_vcm,gallery_loader_vcm,\
        query_loader_vcm,init_loader_vcm = get_data("VCM",args,transform_rgb,transform_thermal,transform_test)

    processed_data = VCM()
    #image-based method
    trainset_gray = VideoDataset_train(processed_data.train_ir,processed_data.train_rgb,1,'video_train',transform_rgb2gray,transform_thermal)
    trainset_gray.train_color_label = processed_data.rgb_label
    trainset_gray.train_thermal_label = processed_data.ir_label
    color_pos_gray, thermal_pos_gray = GenIdx(trainset_gray.train_color_label, trainset_gray.train_thermal_label)

    # add expandable classifier
    model.add_model(num_classes_vcm, init_loader_vcm)
    add_num = sum(model.class_per_task[:-1])

    # Create old frozen model
    old_model = copy.deepcopy(model)
    old_model = old_model.cuda()
    old_model.eval()

    if args.step == 4 and args.resume != "":
        model_path = osp.join(save_model_dir, args.resume)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = 29
            model.load_state_dict(checkpoint['state_dict'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    names.append("vcm")
    test_loaders.append((gallery_loader_vcm, query_loader_vcm))

    cfg.defrost()
    cfg.MAX_EPOCH = 30
    cfg.BASE_LR = 1e-4
    cfg.WARMUP_EPOCHS = 6
    cfg.freeze()
    print(f"==> Stage 4 Strategy: LR={cfg.BASE_LR}, Warmup={cfg.WARMUP_EPOCHS}")

    # Re-initialize the optimizer
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    vis_log_dir = osp.join(args.vis_logs_dir,suffix, "stag4/")
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    writer = SummaryWriter(vis_log_dir)

    trainer = Trainer(model, cfg)
    trainer.update_prototypes(all_prototypes)
    color_pos, thermal_pos = GenIdx(dataset_vcm.train_color_label, dataset_vcm.train_thermal_label)

    for epoch in range(start_epoch, 30):

        current_lr = optimizer.param_groups[0]["lr"]

        sampler = IdentitySampler(dataset_vcm.train_color_label, dataset_vcm.train_thermal_label, color_pos,
                                  thermal_pos, num_pos=args.num_pos, batchSize=args.batch_size)
        dataset_vcm.cIndex = sampler.index1
        dataset_vcm.tIndex = sampler.index2

        if cfg.METHOD == 'PMT':
            if epoch <= cfg.PL_EPOCH:
                sampler_gray = IdentitySampler(trainset_gray.train_color_label, trainset_gray.train_thermal_label,
                                            color_pos_gray, thermal_pos_gray, num_pos=args.num_pos, batchSize=args.batch_size)  # Gray
                # Gray-IR
                trainset_gray.cIndex = sampler_gray.index1
                trainset_gray.tIndex = sampler_gray.index2
                trainloader = data.DataLoader(trainset_gray, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler_gray, drop_last=True)

            else:
                trainloader = data.DataLoader(dataset_vcm, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        else:
            trainloader = data.DataLoader(dataset_vcm, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        trainer.train(epoch, trainloader, optimizer, training_phase=4,
                      add_num=add_num, old_model=old_model, replay=False,writer = writer)

        scheduler.step(epoch)

        if (epoch + 1) % 2 == 0:
            print(f"==> Evaluating Stage 4 (VCM) at epoch {epoch}...")

            current_epoch_cmc_full = []
            current_epoch_r1 = []
            current_epoch_mAP = []

            table = PrettyTable(["Stage 4 Dataset", "Rank-1", "mAP"])

            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP = eval_func2(model, name, test_loader, model.feat_dim, 4, epoch)

                current_epoch_cmc_full.append(cmc)
                current_epoch_r1.append(cmc[0])
                current_epoch_mAP.append(mAP)

                table.add_row([name, f"{cmc[0]:.2%}", f"{mAP:.2%}"])

            avg_r1 = sum(current_epoch_r1) / len(current_epoch_r1)
            avg_mAP = sum(current_epoch_mAP) / len(current_epoch_mAP)

            table.add_row(["-" * 10, "-" * 10, "-" * 10])
            table.add_row(["AVERAGE", f"{avg_r1:.2%}", f"{avg_mAP:.2%}"])

            print(table)

            all_cmc_perstage["stag4"].append(current_epoch_cmc_full)
            all_mAP_perstage["stag4"].append(current_epoch_mAP)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': avg_mAP,
                'rank1': avg_r1,
            }, True, fpath=osp.join(save_model_dir, 'working_checkpoint_step_4.pth.tar'))

        writer.add_scalar("lr", current_lr, epoch)

    del trainloader, trainer, optimizer, init_loader_vcm, dataset_vcm,old_model

    save_checkpoint({
        'state_dict':model.state_dict(),
        'all_cmc_perstage':all_cmc_perstage,
        'all_map_perstage':all_mAP_perstage
    },False,fpath = osp.join(save_model_dir,'last_stage_checkpoint.tar'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual learning for VI-ReID")
    # data
    parser.add_argument(
        "--config_file", default="config/SYSU.yml", help="path to config file", type=str
    )
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--img_h', type=int, default=256, help="input height")
    parser.add_argument('--img_w', type=int, default=128, help="input width")
    parser.add_argument('--num_pos', type=int, default=4)
    parser.add_argument('--batch-size',type = int,default=16)
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')
    parser.add_argument('--mode-sysu',default="all",type = str,
                        help = "all or indoor(test mode for sysu)")
    parser.add_argument('--mode-llcm',default=1,type = int,
                        help = "test mode for llcm(for query)")
    parser.add_argument('--mode-vcm', default=1, type=int,
                        help="test mode for VCM(for query)")

    # optimizer
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--optimizer',type = str,default="AdamW")
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    #Adam
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--lr_pretrain', type=float, default=0.5)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')

    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--step',type = int,default = 0)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join('/mnt/sda1/xyt/datasets/', ''))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--save-model',type = str,default=osp.join(working_dir,'save_model'))
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    parser.add_argument('--vis-logs-dir',type = str,default=osp.join(working_dir,'vis_logs/'))
    args = parser.parse_args()
    main()