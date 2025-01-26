import typing as T
import torch.nn.functional as F
import copy
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from time import time
import argparse
import numpy as np
import pandas as pd
import torch
import torchmetrics
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm.auto import tqdm

from ..dataset import (
    DTIDataModule,
    DTIDataModulecopy,
    DUDEDataModule,
    EnzPredDataModule,
    TDCDataModule,
    TDCDataModulecopy,
    get_task_dir,
)
from ..featurizer import get_featurizer
from ..model import architectures as model_types
from ..model.margin import MarginScheduledLossFunction
from ..utils import config_logger, get_logger, set_random_seed
from ..model.model import FMCADTI
from ..model.config import hyperparameter,FMCAargs
from rdkit import Chem
from ..model.FMCAutils import *
from ..model.DataPrepare import *
from ..featurizer.protein import *

logg = get_logger()


def add_args(parser: ArgumentParser):
    parser.add_argument("--run-id", required=True, help="Experiment ID", dest="run_id")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file",
        default="configs/default_config.yaml",
    )

    # Logging and Paths
    log_group = parser.add_argument_group("Logging and Paths")

    log_group.add_argument(
        "--wandb-proj",
        help="Weights and Biases Project",
        dest="wandb_proj",
    )
    log_group.add_argument(
        "--wandb_save",
        help="Log to Weights and Biases",
        dest="wandb_save",
        action="store_true",
    )
    log_group.add_argument(
        "--log-file",
        help="Log file",
        dest="log_file",
    )
    log_group.add_argument(
        "--model-save-dir",
        help="Model save directory",
        dest="model_save_dir",
    )
    log_group.add_argument(
        "--data-cache-dir",
        help="Data cache directory",
        dest="data_cache_dir",
    )

    # Miscellaneous
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--r", "--replicate", type=int, help="Replicate", dest="replicate"
    )
    misc_group.add_argument(
        "--d", "--device", type=int, help="CUDA device", dest="device"
    )
    misc_group.add_argument(
        "--verbosity", type=int, help="Level at which to log", dest="verbosity"
    )
    misc_group.add_argument(
        "--checkpoint", default=None, help="Model weights to start from"
    )

    # Task and Dataset
    task_group = parser.add_argument_group("Task and Dataset")

    task_group.add_argument(
        "--task",
        choices=[
            "biosnap",
            "bindingdb",
            "davis",
            "biosnap_prot",
            "biosnap_mol",
            "dti_dg",
        ],
        type=str,
        help="Task name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol.",
    )
    task_group.add_argument(
        "--contrastive-split",
        type=str,
        help="Contrastive split",
        dest="contrastive_split",
        choices=["within", "between"],
    )

    # Model and Featurizers
    model_group = parser.add_argument_group("Model and Featurizers")

    model_group.add_argument(
        "--drug-featurizer", help="Drug featurizer", dest="drug_featurizer"
    )
    model_group.add_argument(
        "--target-featurizer", help="Target featurizer", dest="target_featurizer"
    )
    model_group.add_argument(
        "--model-architecture", help="Model architecture", dest="model_architecture"
    )
    model_group.add_argument(
        "--latent-dimension", help="Latent dimension", dest="latent_dimension"
    )
    model_group.add_argument(
        "--latent-distance", help="Latent distance", dest="latent_distance"
    )
    # Training
    train_group = parser.add_argument_group("Training")

    train_group.add_argument("--epochs", type=int, help="number of total epochs to run")
    train_group.add_argument("-b", "--batch-size", type=int, help="batch size")
    train_group.add_argument(
        "--cb", "--contrastive-batch-size", type=int, help="contrastive batch size"
    )
    train_group.add_argument("--shuffle", type=bool, help="shuffle data")
    train_group.add_argument("--num-workers", type=int, help="number of workers")
    train_group.add_argument("--every-n-val", type=int, help="validate every n epochs")

    train_group.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        help="initial learning rate",
        dest="lr",
    )
    train_group.add_argument(
        "--lr-t0", type=int, help="number of epochs to reset learning rate"
    )
    train_group.add_argument(
        "--contrastive", type=bool, help="run contrastive training"
    )
    train_group.add_argument(
        "--clr", type=float, help="initial learning rate", dest="clr"
    )
    train_group.add_argument(
        "--clr-t0", type=int, help="number of epochs to reset learning rate"
    )
    train_group.add_argument(
        "--margin-fn", type=str, help="margin function", dest="margin_fn"
    )
    train_group.add_argument(
        "--margin-max", type=float, help="margin max", dest="margin_max"
    )
    train_group.add_argument(
        "--margin-t0", type=int, help="number of epochs to reset margin"
    )
    train_group.add_argument(
        "--use-sigmoid-cosine", action="store_true", dest="sigmoid_cosine",
        help="Use sigmoid cosine distance instead of just cosine distance for contrastive loss"
    )

    return parser


absolute_path = Path(f"/home/liujin/data/ConPLex-main/sum/tdc_pt50train.txt")
predictions=[]




def test(model, data_generator, metrics, device=None, classify=True):
    if device is None:
        device = torch.device("cpu")

    metric_dict = {}

    for k, met_class in metrics.items():
        if classify:
            met_instance = met_class(task="binary")
        else:
            met_instance = met_class()
        met_instance.to(device)
        met_instance.reset()
        metric_dict[k] = met_instance

    model.eval()

    for _, batch in tqdm(enumerate(data_generator), total=len(data_generator)):
        pred, label = step(model, batch, device)
        predictions.append({
        'pred': pred.tolist(),
        'label':label
    })
        # 将结果转换为DataFrame
        predictions_df = pd.DataFrame(predictions)


        if classify:
            label = label.int()
        else:
            label = label.float()

        for _, met_instance in metric_dict.items():
            met_instance(pred, label)

    #  将结果保存到CSV文件
    predictions_df.to_csv(absolute_path, index=False)

    results = {}
    for k, met_instance in metric_dict.items():
        res = met_instance.compute()
        results[k] = res

    for met_instance in metric_dict.values():
        met_instance.to("cpu")

    return results

def step(model, batch, device=None):
    if device is None:
        device = torch.device("cpu")

    drug, target, label = batch  # target is (D + N_pool)
    pred = model(drug.to(device), target.to(device))
    label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

    return pred, label


def contrastive_step(model, batch, device=None):
    if device is None:
        device = torch.device("cpu")

    anchor, positive, negative = batch

    anchor_projection = model.target_projector(anchor.to(device))
    positive_projection = model.drug_projector(positive.to(device))
    negative_projection = model.drug_projector(negative.to(device))

    return anchor_projection, positive_projection, negative_projection


def wandb_log(m, do_wandb=True):
    if do_wandb:
        wandb.log(m)


# args_f = FMCAargs()
# hp = hyperparameter()
# seed=42

# def load_teacher_model(hp, args, model_path, device):
#     # 初始化教师模型
#     teacher_model = FMCADTI(hp, args_f)
    
#     # 加载模型权重
#     teacher_model.load_state_dict(torch.load(model_path))
    
#     # 将模型移动到指定设备
#     teacher_model.to(device)
    
#     return teacher_model

# def load_teacher_model(hp, args_f, model_path, device):
#     teacher_model = FMCADTI(hp, args_f).to(device)
#     state_dict = torch.load(model_path, map_location=device)

#     # 过滤掉适配层的权重
#     filtered_state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('drug_adapt') or k.startswith('protein_adapt'))}
    
#     # 加载过滤后的状态字典
#     teacher_model.load_state_dict(filtered_state_dict, strict=False)
    
#     return teacher_model

def main(args):
    logg.info(" there is Initializing knowledge_distillation")
    # delattr(args, "main_func")
    config = OmegaConf.load(args.config)
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(arg_overrides)

    save_dir = f'{config.get("model_save_dir", ".")}/{config.run_id}'
    os.makedirs(save_dir, exist_ok=True)

    # Logging
    if "log_file" not in config:
        config.log_file = None
    else:
        os.makedirs(Path(config.log_file).parent, exist_ok=True)
    config_logger(
        config.log_file,
        "%(asctime)s [%(levelname)s] %(message)s",
        config.verbosity,
        use_stdout=True,
    )

    # Set CUDA device
    device_no = 5
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    logg.info(f"Using CUDA device {device}")

    # Set random state
    logg.debug(f"Setting random state {config.replicate}")
    set_random_seed(config.replicate)

    # Load DataModule
    logg.info("Preparing DataModule")
    task_dir = get_task_dir(config.task, database_root=config.data_cache_dir)

    drug_featurizer = get_featurizer(config.drug_featurizer, save_dir=task_dir)
    target_featurizer = get_featurizer(config.pt50_target_featurizer, save_dir=task_dir)

    if config.task == "dti_dg":
        config.classify = False
        config.latent_activation = "GELU"
        config.watch_metric = "val/pcc"
        datamodule = TDCDataModulecopy(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            seed=config.replicate,
            batch_size=config.batch_size,
            shuffle=config.shuffle_dti,#测试集不打乱
            num_workers=config.num_workers,
        )
    elif config.task in EnzPredDataModule.dataset_list():
        config.classify = True
        config.latent_activation = "ReLU"
        config.watch_metric = "val/aupr"
        datamodule = EnzPredDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            seed=config.replicate,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
    else:
        config.classify = True
        config.watch_metric = "val/aupr"
        config.latent_activation = "ReLU"
        datamodule = DTIDataModulecopy(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            seed=config.replicate,
            batch_size=config.batch_size,
            shuffle=config.shuffle_dti,
            num_workers=config.num_workers,
        )
    datamodule.prepare_data()
    datamodule.setup()

    # Load DataLoaders
    logg.info("Getting DataLoaders")
    training_generator = datamodule.train_dataloader()
    # validation_generator = datamodule.val_dataloader()
    # testing_generator = datamodule.test_dataloader()


    # args_f['max_drug_seq'] = 20
    # args_f['max_protein_seq'] = 242
    # args_f['input_d_dim'] = 292378
    # args_f['input_p_dim'] = 209
    # args_f['d_channel_size'] = [20, 128, 256, 512]  # 这里的第一个元素是 max_drug_seq
    # args_f['p_channel_size'] = [242, 128, 256, 512]

    # #对比学习训练  替换成我们的二分类网络
    # if config.knowledge_distillation:
    #     logg.info("Loading contrastive data (DUDE)")
    #     dude_drug_featurizer = get_featurizer(
    #         config.drug_featurizer,
    #         save_dir=get_task_dir("DUDe", database_root=config.data_cache_dir),
    #     )

    #     dude_target_featurizer = get_featurizer(
    #         config.target_featurizer,
    #         save_dir=get_task_dir("DUDe", database_root=config.data_cache_dir),
    #     )

    #     contrastive_datamodule = DUDEDataModule(
    #         get_task_dir("DUDe", database_root=config.data_cache_dir),
    #         args_f,
    #         hp,
    #         config.contrastive_split,
    #         dude_drug_featurizer,
    #         dude_target_featurizer,
    #         device=device,
    #         batch_size=config.contrastive_batch_size,
    #         shuffle=config.shuffle,
    #         num_workers=config.num_workers,
    #         drop_last=True,  # 确保 drop_last 参数一致
    #         seed=seed,
    #         )

    #     contrastive_datamodule.prepare_data()
    #     contrastive_datamodule.setup(stage="fit")
    #     contrastive_generator = contrastive_datamodule.train_dataloader()
    #     t_train_loader = contrastive_datamodule.t_train_dataloader()

    config.drug_shape = drug_featurizer.shape
    config.target_shape = target_featurizer.shape

    # Model
    logg.info("Initializing model")
    model = getattr(model_types, config.model_architecture)(
        config.drug_shape,
        config.target_shape,
        latent_dimension=config.latent_dimension,
        latent_distance=config.latent_distance,
        latent_activation=config.latent_activation,
        classify=config.classify,
    )
    if "checkpoint" in config:
        state_dict = torch.load(config.checkpoint)
        model.load_state_dict(state_dict)

    model = model.to(device)
    logg.info(model)


    # 加载 model_max 的状态字典
    model_path = "/home/liujin/data/ConPLex-main/best_models/TestRun/TestRunProtT5XLUniref50Featurizer_resnet_9_best_model.pt"  # 替换为你的模型文件路径
    model_state_dict = torch.load(model_path)

    # 将状态字典应用到你的模型实例上
    model.load_state_dict(model_state_dict)



    # # Optimizers
    # logg.info("Initializing optimizers")
    # opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     opt, T_0=config.lr_t0
    # )

    # #对比学习
    # logg.info("Initializing knowledge_distillation")
    # if config.knowledge_distillation:
    #     config.dist_fn = "sigmoid_cosine_distance" if config.sigmoid_cosine else "cosine_distance"
    #     contrastive_loss_fct = MarginScheduledLossFunction(
    #         M_0=config.margin_max,
    #         N_epoch=config.epochs,
    #         N_restart=config.margin_t0,
    #         update_fn=config.margin_fn,
    #         dist_fn=config.dist_fn,
    #     )
    #     opt_contrastive = torch.optim.AdamW(model.parameters(), lr=config.clr)
    #     lr_scheduler_contrastive = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #         opt_contrastive, T_0=config.clr_t0
    #     )

    # Metrics
    logg.info("Initializing metrics")
    max_metric = 0
    # model_max = copy.deepcopy(model)

    if config.task == "dti_dg":
        loss_fct = torch.nn.MSELoss()
        val_metrics = {
            "val/mse": torchmetrics.MeanSquaredError,
            "val/pcc": torchmetrics.PearsonCorrCoef,
        }

        test_metrics = {
            "test/mse": torchmetrics.MeanSquaredError,
            "test/pcc": torchmetrics.PearsonCorrCoef,
        }
    else:
        loss_fct = torch.nn.BCELoss()
        val_metrics = {
            "val/aupr": torchmetrics.AveragePrecision,
            "val/auroc": torchmetrics.AUROC,
        }

        test_metrics = {
            "test/aupr": torchmetrics.AveragePrecision,
            "test/auroc": torchmetrics.AUROC,
        }

    # Initialize wandb
    do_wandb = config.wandb_save and ("wandb_proj" in config)
    if do_wandb:
        logg.info(f"Initializing wandb project {config.wandb_proj}")
        wandb.init(
            project=config.wandb_proj,
            name=config.run_id,
            config=dict(config),
        )
        wandb.watch(model, log_freq=100)
    # logg.info("Config:")
    # logg.info(json.dumps(dict(config), indent=4))

    logg.info("Beginning Training")

    torch.backends.cudnn.benchmark = True

    # # model_path = '/home/liujin/data/ConPLex-main/models/valid_best_checkpoint.pth'  # 模型权重文件的路径

    # # Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    # # decompose = "bcm"
    # # decompose_protein = "category"
    # # SEED=114514
    # # K_Fold=5
    # # split_random = True

    # # root_path=get_task_dir("DUDe", database_root=config.data_cache_dir)

    # # train_data_list = pd.read_csv(root_path / 'train.txt', sep=' ', header=None)

    # # test_data_list = pd.read_csv(root_path /'test.txt', sep=' ', header=None)

    # # # 将 DataFrame 转换为列表
    # # train_data_list = train_data_list.values.tolist()
    # # test_data_list = test_data_list.values.tolist()

    # # train_data_list = shuffle_dataset(train_data_list, SEED)
    # # test_data_list = shuffle_dataset(test_data_list, SEED)

    # # train_dataset, valid_dataset = get_kfold_data(
    # #         1, train_data_list, k=K_Fold)
    
    # # trainSmiles, trainProtein, trainLabel, \
    # #         valSmiles, valProtein, valLabel, \
    # #         testSmiles, testProtein, testLabel, \
    # #         frag_set_d, frag_set_p, \
    # #         frag_len_d, frag_len_p, \
    # #         words2idx_d, words2idx_p = load_frag(train_dataset, valid_dataset, test_data_list, decompose,
    # #                                                            decompose_protein, unseen_smiles=False, k=3,
    # #                                                            split_random=split_random)
    # # n=3
    # # args_f['max_drug_seq'] = 20
    # # args_f['max_protein_seq'] = 242
    # # args_f['input_d_dim'] = 292378
    # # args_f['input_p_dim'] = 209
    # # args_f['d_channel_size'] = [20, 128, 256, 512]  # 这里的第一个元素是 max_drug_seq
    # # args_f['p_channel_size'] = [242, 128, 256, 512]  # 这里的第一个元素是 max_protein_seq
    # # args_f['max_drug_seq'] = 2048
    # # args_f['max_protein_seq'] = 1024
    # # args_f['input_d_dim'] = 292378
    # # args_f['input_p_dim'] = 209
    # # args_f['d_channel_size'] = [2048, 128, 256, 512]  # 这里的第一个元素是 max_drug_seq
    # # args_f['p_channel_size'] = [1024, 128, 256, 512]  # 这里的第一个元素是 max_protein_seq



    # # # 加载预训练的教师模型
    # # teacher_model = load_teacher_model(hp, args_f, model_path, device)
    # # teacher_model=teacher_model.to(device)

    # # total_k_loss = 0.0

    # torch.cuda.empty_cache()


    # # Begin Training 训练的地方
    # start_time = time()
    # for epo in range(config.epochs):
    #     model.train()
    #     epoch_time_start = time()

    #     # Main Step
    #     for i, batch in tqdm(
    #         enumerate(training_generator), total=len(training_generator)
    #     ):
    #         pred, label = step(model, batch, device)  # batch is (2048, 1024, 1)

    #         loss = loss_fct(pred, label)

    #         wandb_log(
    #             {
    #                 "train/step": (epo * len(training_generator) * config.batch_size)
    #                 + (i * config.batch_size),
    #                 "train/loss": loss,
    #             },
    #             do_wandb,
    #         )

    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #     lr_scheduler.step()

    #     wandb_log(
    #         {
    #             "epoch": epo,
    #             "train/lr": lr_scheduler.get_lr()[0],
    #         },
    #         do_wandb,
    #     )
    #     logg.info(
    #         f"Training at Epoch {epo + 1} with loss {loss.cpu().detach().numpy():8f}"
    #     )
    #     logg.info(f"Updating learning rate to {lr_scheduler.get_lr()[0]:8f}")

    #     # assert len(contrastive_generator) == len(t_train_loader), "Data loaders have different lengths!"

    #     # if config.knowledge_distillation:
    #     #     logg.info(f"Training knowledge_distillation at Epoch {epo + 1}")
    #     #     for i, batch1 in tqdm(enumerate(contrastive_generator), total=len(contrastive_generator)):
    #     #         student_pred, student_label = step(model, batch1, device)

    #     #         # batch2 = {k: v.to(device) for k, v in batch2.items()}

    #     #         # drug, target, label_ = batch2



    #     #         # drug = drug.to(device)
    #     #         # target = target.to(device)

    #     #         # # 获取教师模型的输出
    #     #         # with torch.no_grad():
                   
    #     #         #     teacher_pred = teacher_model(drug, target)

    #     #         # # 打印 student_pred 的形状
    #     #         # print("Shape of student_pred:", student_pred.shape)
                        

    #     #         # 计算知识蒸馏损失
                
    #     #         # kd_loss = F.kl_div(
    #     #         #         F.log_softmax(student_pred / config.temperature, dim=0),
    #     #         #         F.softmax(teacher_pred / config.temperature, dim=0),
    #     #         #         reduction='batchmean'
    #     #         #     ) * (config.temperature ** 2)
                

    #     #         s_loss = loss_fct(student_pred, student_label)

    #     #         # # 总损失
    #     #         # k_loss = (1 - config.alpha) * s_loss + config.alpha * kd_loss

    #     #         s_loss.backward()
    #     #         # # contrastive_loss = contrastive_loss_fct(drug, target, label )

    #     #         # # 累积损失
    #     #         # total_k_loss += k_loss.item()

    #     #         # 日志记录
    #     #         wandb_log(
    #     #             {
    #     #                 "train/c_step": (
    #     #                     epo
    #     #                     * len(training_generator)
    #     #                     * config.contrastive_batch_size
    #     #                 )
    #     #                 + (i * config.contrastive_batch_size),
    #     #                 # "train/kd_loss": kd_loss.item(),
    #     #                 # "train/s_loss": s_loss.item(),
    #     #                 # "train/total_loss": k_loss.item(),
    #     #             },
    #     #             do_wandb,
    #     #         )

    #     #         opt_contrastive.zero_grad()
    #     #         # k_loss.backward()
    #     #         opt_contrastive.step()

            
    #     #     lr_scheduler_contrastive.step()

    #     #     # 计算平均损失
    #     #     avg_k_loss = total_k_loss / len(contrastive_generator)

    #     #     # 日志记录
    #     #     wandb_log(
    #     #         {
    #     #             "epoch": epo,
    #     #             "train/avg_loss": avg_k_loss,
    #     #             "train/contrastive_lr": lr_scheduler_contrastive.get_last_lr(),
    #     #         },
    #     #         do_wandb,
    #     #     )

    #     #     logg.info(
    #     #         f"Training at Knowledge Distillation Epoch {epo + 1} with avg loss {avg_k_loss:8f}"
    #     #     )
    #     #     logg.info(
    #     #         f"Updating contrastive learning rate to {lr_scheduler_contrastive.get_last_lr()[0]:8f}"
    #     #     )
            

    #     epoch_time_end = time()

    #     # Validation
    #     if epo % config.every_n_val == 0:
    #         with torch.set_grad_enabled(False):
    #             val_results = test(
    #                 model,
    #                 validation_generator,
    #                 val_metrics,
    #                 device,
    #                 config.classify,
    #             )

    #             val_results["epoch"] = epo
    #             val_results["Charts/epoch_time"] = (
    #                 epoch_time_end - epoch_time_start
    #             ) / config.every_n_val

    #             wandb_log(val_results, do_wandb)

    #             if val_results[config.watch_metric] > max_metric:
    #                 logg.debug(
    #                     f"Validation AUPR {val_results[config.watch_metric]:8f} > previous max {max_metric:8f}"
    #                 )
    #                 model_max = copy.deepcopy(model)
    #                 max_metric = val_results[config.watch_metric]
    #                 model_save_path = Path(
    #                     f"{save_dir}/{config.run_id}{config.target_featurizer}_best_model_epoch{epo:02}.pt"
    #                 )
    #                 torch.save(
    #                     model_max.state_dict(),
    #                     model_save_path,
    #                 )
    #                 logg.info(f"Saving checkpoint model to {model_save_path}")

    #                 if do_wandb:
    #                     art = wandb.Artifact(f"dti-{config.run_id}", type="model")
    #                     art.add_file(model_save_path, model_save_path.name)
    #                     wandb.log_artifact(art, aliases=["best"])

    #             logg.info(f"Validation at Epoch {epo + 1}")
    #             for k, v in val_results.items():
    #                 if not k.startswith("_"):
    #                     logg.info(f"{k}: {v}")

    # end_time = time()


    # Testing
    logg.info("Beginning testing")
    try:
        with torch.set_grad_enabled(False):
            model = model.eval()

            test_start_time = time()
            test_results = test(
                model,
                training_generator,
                test_metrics,
                device,
                config.classify,
            )
            test_end_time = time()



            # test_results["epoch"] = epo + 1
            test_results["test/eval_time"] = test_end_time - test_start_time
            # test_results["Charts/wall_clock_time"] = end_time - start_time
            wandb_log(test_results, do_wandb)

            logg.info("Final Testing")
            for k, v in test_results.items():
                if not k.startswith("_"):
                    logg.info(f"{k}: {v}")

            if do_wandb:
                art = wandb.Artifact(f"dti-{config.run_id}", type="model")
                wandb.log_artifact(art, aliases=["best"])

    except Exception as e:
        logg.error(f"Testing failed with exception {e}")




if __name__ == "__main__":
    # best_model = main()
    parser = argparse.ArgumentParser(description="dti")
    parser = add_args(parser) 
    args = parser.parse_args() 
    best_model = main(args)
