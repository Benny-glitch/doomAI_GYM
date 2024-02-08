import local.agent as game_agent
import zipfile
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import utils
from backend.basic_agent.vizDoomGymEnv import VizDoomGym


def train_file_union():
    # Percorso del file ZIP
    zip_file_path_1 = "dockerVolume/model_50000_1.zip"
    zip_file_path_2 = "dockerVolume/model_50000_2.zip"

    # Percorso di destinazione per l'estrazione dei file
    extract_path_1 = "extracted_model/best_model_50000_1"
    extract_path_2 = "extracted_model/best_model_50000_2"

    # Estrai il file ZIP
    with zipfile.ZipFile(zip_file_path_1, 'r') as zip_ref:
        zip_ref.extractall(extract_path_1)

    # Estrai il file ZIP
    with zipfile.ZipFile(zip_file_path_2, 'r') as zip_ref:
        zip_ref.extractall(extract_path_2)

    # Carica il modello PyTorch
    first_model = torch.load(os.path.join(extract_path_1, "policy.pth"))

    # Carica altri modelli PyTorch
    second_model = torch.load(os.path.join(extract_path_2, "policy.pth"))

    for chiave in first_model.keys():
        first_model[chiave] = (first_model[chiave] + second_model[chiave]) / 2  # Media semplice

    # Salva il modello combinato
    torch.save(first_model, "combined_model.pth")

    # Carica i pesi combinati
    combined_weight = torch.load("combined_model.pth")

    # Inizializza il modello PPO utilizzando la tua Policy Network personalizzata
    original_env = DummyVecEnv([lambda: VizDoomGym(render=True, scenarios=utils.BASIC_CONFIG_PATH)])
    ppo_model = PPO('CnnPolicy', original_env)
    # Carica i pesi combinati nel modello PPO
    ppo_model.policy.load_state_dict(combined_weight)

    game_agent.agent_init(ppo_model)