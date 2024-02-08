import os

BASIC_CONFIG_PATH = os.path.dirname(os.path.abspath(__file__)) + '/doomGame/ViZDoom/scenarios/basic.cfg'

DEADLY_CONFIG_PATH = os.path.dirname(os.path.abspath(__file__)) + '/doomGame/ViZDoom/scenarios/deadly_corridor.cfg'

TRAIN_PATH_BASIC = os.path.dirname(os.path.abspath(__file__)) + '/train_models/train_basic/'

TRAIN_PATH_DEADLY = os.path.dirname(os.path.abspath(__file__)) + '/train/train_deadly/'

LOG_DIR_REMOTE = '/usr/share/app/logs/log_basic'

MODEL_DIR_REMOTE = '/usr/share/app/train/train_basic'

LOCAL_DIR_LOGS = '../../logs/log_basic'

LOCAL_DIR_MODELS = '../../train/train_basic'

combined_models = os.path.dirname(os.path.abspath(__file__)) + '/modello_combinato.pth'


