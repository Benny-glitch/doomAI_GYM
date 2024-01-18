from colorama import Fore, Style, init
from termcolor import colored
from pyfiglet import Figlet
import six

def log(entry_num, string, color, figlet_font):
    if figlet_font is None:
        six.print_(colored(string, color))
    else:
        f = Figlet(font=figlet_font)
        six.print_(colored(f.renderText(string), color))


def options_menu():
    log("Doom AI Game","blue","slant")
    log("Benvenuto a Doom AI Game", "light_green", None)






if __name__ == "__main__":
    options_menu()
