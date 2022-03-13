import os.path as osp


CURRENT_PATH = osp.dirname(osp.realpath(__file__))

class SystemConfig:

    root_dir = osp.realpath(osp.join(CURRENT_PATH, ".."))
    data_dir = osp.join(root_dir, "data")
    model_dir = osp.join(root_dir, "models")


system_config = SystemConfig()
