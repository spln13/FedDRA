import torch

from server import server
from client import client





def do():
    # 联邦学习主流程
    pass



def main():
    # server.init()
    # init clients
    # 这里定义一些联邦学习的参数
    client_nums = 20
    model_name = 'MiniVGG'
    dataset = 'cifar10'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ppo related
    d_glob = 6
    d_cli = 7
    p_low = 0.2
    p_high = 0.9
    E_min = 1
    E_max = 5
    hidden = 256

    # init server


    pass

