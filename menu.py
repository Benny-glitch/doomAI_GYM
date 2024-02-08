from termcolor import colored
from pyfiglet import Figlet
import six

import utils
import subprocess


def log(string, color, figlet_font):
    if figlet_font is None:
        six.print_(colored(string, color))
    else:
        f = Figlet(font=figlet_font)
        six.print_(colored(f.renderText(string), color))


def options_menu():
    log("Doom AI Game", "red", "slant")
    log("Benvenuto a Doom AI Game", "dark_grey", None)
    log("1. Lascia giocare l'AI", "white", None)
    log("2. Traina l'IA", "white", None)
    log("3. Lascia giocare l'IA da remoto", "white", None)
    log("4. Carica un modello di training diverso", "white", None)
    choice = input()
    if int(choice) == 1:
        play_ai()
    elif int(choice) == 2:
        train_ai()
    elif int(choice) == 3:
        train_ai_remote()
    elif int(choice) == 4:
        load_different_model()


def play_ai():
    invalid_opt = True
    log("1. BASIC ENVIRONMENT", "dark_grey", None)
    log("2. DEADLY CORRIDOR ENVIRONMENT", "dark_grey", None)
    choice = input()
    while invalid_opt:
        if int(choice) == 1:
            invalid_opt = False
            agent.agent_init(utils.BASIC_CONFIG_PATH)
        if int(choice) == 2:
            invalid_opt = False
            agent.agent_init(utils.DEADLY_CONFIG_PATH)


def train_ai():
    invalid_opt = True
    log("1. BASIC ENVIRONMENT", "dark_grey", None)
    log("2. DEADLY CORRIDOR ENVIRONMENT", "dark_grey", None)
    choice = input()
    while invalid_opt:
        if int(choice) == 1:
            print("Inserisci il numero di step totali da effettuare: ")
            tot_step = input()
            print("Inserisci la frequenza di salvataggio: ")
            freq = input()
            invalid_opt = False
            game_train_basic.start_training(tot_steps=int(tot_step),feq_saving=int(freq))
        if int(choice) == 2:
            print("Inserisci il numero di step totali da effettuare: ")
            tot_step = input()
            print("Inserisci la frequenza di salvataggio: ")
            freq = input()
            game_train_deadly.start_training(tot_steps=tot_step, feq_saving=freq)
            invalid_opt = False


def train_ai_remote():
    # Percorso del file docker-compose.yml
    compose_file = "docker-compose.yml"

    # Esegui il comando docker-compose up
    processo = subprocess.Popen(["docker-compose", "-f", compose_file, "up", "-d"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    stdout, stderr = processo.communicate()

    # Stampa l'output e l'eventuale errore
    print("Output:", stdout.decode())


def load_different_model():
    pass


if __name__ == "__main__":
    options_menu()

