from termcolor import colored
from pyfiglet import Figlet
import six

def log(string, color, figlet_font):
    if figlet_font is None:
        six.print_(colored(string, color))
    else:
        f = Figlet(font=figlet_font)
        six.print_(colored(f.renderText(string), color))


def options_menu():
    log("Doom AI Game","red","slant")
    log("Benvenuto a Doom AI Game", "dark_grey", None)
    log("1. Lascia giocare l'AI", "white", None)
    log("2. Traina l'IA", "white", None)
    log("3. Lascia giocare l'IA da remoto","white", None)
    log("4. Carica un modello di training diverso","white", None)

def play_AI():
    pass

def train_AI():
    pass

def play_AI_remte():
    pass

def load_different_model():
    pass


if __name__ == "__main__":
    options_menu()
