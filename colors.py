
class ColorfulPrinter(object):
    # Regular Colors
    Black = '\033[0;30m'  # Black
    Red = '\033[0;31m'  # Red
    Green = '\033[0;32m'  # Green
    Yellow = '\033[0;33m'  # Yellow
    Blue = '\033[0;34m'  # Blue
    Purple = '\033[0;35m'  # Purple
    Cyan = '\033[0;36m'  # Cyan
    White = '\033[0;37m'  # White

    # Bold
    BBlack = '\033[1;30m'  # Black
    BRed = '\033[1;31m'  # Red
    BGreen = '\033[1;32m'  # Green
    BYellow = '\033[1;33m'  # Yellow
    BBlue = '\033[1;34m'  # Blue
    BPurple = '\033[1;35m'  # Purple
    BCyan = '\033[1;36m'  # Cyan
    BWhite = '\033[1;37m'  # White

    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'  # Red
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def colorful_string(self, str, *colors):
        for col in colors:
            str = col + str + self.ENDC
        return str

    def print_colored(self, str, *colors):
        print self.colorful_string(str, *colors)
