import time
import zipfile
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import utils
from backend.basic_agent.vizDoomGymEnv import VizDoomGym

# Percorso del file ZIP
zip_file_path = "train_models/train_basic/best_model_100000.zip"

# Percorso di destinazione per l'estrazione dei file
extract_path_100000 = "extracted_model/best_model_100000"

# Estrai il file ZIP
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path_100000)

zip_file_path = "train_models/train_basic/best_model_90000.zip"

# Percorso di destinazione per l'estrazione dei file
extract_path_90000 = "extracted_model/best_model_90000"

# Estrai il file ZIP
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path_90000)

# Carica il modello PyTorch
modello = torch.load(os.path.join(extract_path_100000, "policy.pth"))

# Carica altri modelli PyTorch
altro_modello_1 = torch.load(os.path.join(extract_path_90000, "policy.pth"))

for chiave in modello.keys():
    modello[chiave] = (modello[chiave] + altro_modello_1[chiave]) / 2  # Media semplice

# Salva il modello combinato
torch.save(modello, "modello_combinato.pth")

# Carica i pesi combinati
pesi_combinati = torch.load("modello_combinato.pth")

# Inizializza il modello PPO utilizzando la tua Policy Network personalizzata
original_env = DummyVecEnv([lambda: VizDoomGym(render=True, scenarios=utils.BASIC_CONFIG_PATH)])
modello_ppo = PPO('CnnPolicy', original_env)
# Carica i pesi combinati nel modello PPO
modello_ppo.policy.load_state_dict(pesi_combinati)

for episode in range(100):
    obs = original_env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = modello_ppo.predict(obs)
        obs, reward, done, info = original_env.step(action)
        time.sleep(0.02)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(total_reward, episode))
    time.sleep(0.5)

