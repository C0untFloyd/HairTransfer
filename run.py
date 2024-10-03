import globals
from hair_swap import HairFast, get_parser
from ui.main import render_ui


if __name__ == '__main__':
    globals.hair_fast = HairFast(get_parser().parse_args([]))
    ui = render_ui()
    ui.launch(inbrowser="True")
