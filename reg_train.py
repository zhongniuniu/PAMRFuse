#!/usr/bin/python3

import argparse
import os
from reg_trainer.mytrain import Cyc_Trainer
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()     #创建一个参数解析实例
    parser.add_argument('--config', type=str, default='reg_Yaml/reg.yaml', help='Path to the config file.') #载入初始参数文件
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'reg':
        trainer = Cyc_Trainer(config)   #初始化模型

    trainer.train()  #调用训练函数
    
    



###################################
if __name__ == '__main__':
    main()