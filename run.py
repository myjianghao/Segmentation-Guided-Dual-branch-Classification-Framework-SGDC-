import os
import argparse
import yaml
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from model.Resnet import *
# from model.Unet import U_Net
from model.Res_Unet import U_Res50
# from model.mobilenetv3 import mobilenet_v3_small,mobilenet_v3_large
# from model.inception import inception_v3
# from model.vgg import vgg19
# from model.CSFT import CSFT_Create
# from model.hubconf import ghostnet_1x
# from model.Res_bert import *
from experiment import *
from dataset import *
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model for fat classification')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar="FILE",
                        help="path to the config file",
                        default="./config/resnet.yaml")
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['logging_params']['name'])
    num_classes = 4
    if config['model_params']['is_regression']:
        num_classes = 1
    model = eval(config['model_params']['name'])(pretrained=config['model_params']['pretrained'],
                                                 num_classes=num_classes)
    with open(config["data_params"]["data_path"], 'r') as file:
        try:
            file_dict = json.load(file)
        except Exception as exc:
            print(exc)
    """with open('mask_filename.json', 'r') as file:
        try:
            mask_dict = json.load(file)
        except Exception as exc:
            print(exc)"""
    loss_array = np.zeros((4, len(file_dict) - config["data_params"]["val_size"] - config["data_params"][
        "test_size"]))
    loss_array[:] = np.nan
    experiment = eval(config['exp_params']['name'])(model, loss_array, config["data_params"]["train_batch_size"],
                                                    config["data_params"]["mixup"],
                                                    config["model_params"]["is_regression"],
                                                    config["model_params"]["is_camloss"], config['exp_params'])
    dataset_loader = eval(config['data_params']['name'])(**config["data_params"], file_dict=file_dict,
                                                         is_regression=config["model_params"]["is_regression"],
                                                         pin_memory=0)
    # dataset_loader.setup()
    if config['running_type'] == 'train':
        runner = Trainer(logger=tb_logger,
                         callbacks=[
                             LearningRateMonitor(),
                             ModelCheckpoint(save_top_k=2,
                                             dirpath=os.path.join("./checkpoints",
                                                                  config['model_params']['name'],
                                                                  tb_logger.log_dir[len(tb_logger.root_dir) + 1:]),
                                             monitor="val_accuracy",
                                             save_weights_only=True,
                                             mode="max",
                                             save_last=True),
                         ],
                         strategy=DDPPlugin(find_unused_parameters=False),
                         **config['trainer_params'])

        print(f"======= Training {config['model_params']['name']} =======")
        runner.fit(experiment, datamodule=dataset_loader)
    if config['running_type'] == 'test':
        # 加载训练好的模型
        filename = os.listdir(os.path.join("./checkpoints",
                                           config['model_params']['name'],
                                           config['version_type']))
        runner = Trainer(logger=tb_logger, gpus=0)
        print(f"======= Testing {config['model_params']['name']} =======")
        print("\n", os.path.join("./checkpoints", config['model_params']['name'],
                                 config['version_type'],
                                 filename[config['load_index']]), "\n")
        runner.test(experiment, ckpt_path=os.path.join("./checkpoints", config['model_params']['name'],
                                                       config['version_type'],
                                                       filename[config['load_index']]),
                    datamodule=dataset_loader)
