# The following file doesn't execute any task, except configuring the tools.
from pyutils import Colors, RGBColor, TraceBackColor, progress, Color
from pyutils.progress import format_total, format_dl_eta, format_time_per_step, format_sep, format_added_values
import sys, os

# Nord theme for True color terminals. Uncomment to activate
# Colors.set_theme(
#     accent=RGBColor(143, 188, 187),  # Frost - Teal
#     text=RGBColor(216, 222, 233),  # Snow Storm - Main text
#     error=RGBColor(220, 70, 80),  # Flashy Error - Bright Crimson
#     warning=RGBColor(255, 170, 50),  # Flashy Warning - Golden Orange
#     success=RGBColor(100, 220, 120),  # Flashy Success - Emerald Green
#     link=RGBColor(136, 192, 208),  # Frost - Light Blue
#     white=RGBColor(236, 239, 244),  # Snow Storm - Bright text
#     black=RGBColor(46, 52, 64),  # Polar Night - Dark background
#     red=RGBColor(191, 97, 106),  # Aurora - Red
#     blue=RGBColor(94, 129, 172),  # Frost - Dark Blue
#     green=RGBColor(163, 190, 140),  # Aurora - Green
#     yellow=RGBColor(235, 203, 139),  # Aurora - Yellow
#     cyan=RGBColor(143, 188, 187),  # Frost - Teal
#     magenta=RGBColor(180, 142, 173),  # Aurora - Purple
#     brown=RGBColor(121, 85, 72),  # Invented Brown
#     orange=RGBColor(208, 135, 112),  # Aurora - Orange
#     purple=RGBColor(180, 142, 173),  # Aurora - Purple
#     pink=RGBColor(216, 155, 176)  # Invented Pink
# )
# Configure the traceback formatting tool
sys.excepthook = TraceBackColor()

# Configure the deep learning progress bar
progress.set_config(
    # done_color=Colors.darken,
    type="dl",
    cursors=(f"{Color(8)}╺", f"╸{Color(8)}"),
    cu="━",
    cd="━",
    max_width=40,
    # refresh_rate=0.01,
    ignore_term_width="PYCHARM_HOSTED" in os.environ,
    delim=(f" {Color(197)}", f"{Colors.reset}"),
    done_delim=(f" {Color(10)}", f"{Colors.reset}"),
    done_charac=f"━",
    end="",
    post_cb=(
            format_total,
            format_dl_eta,
            format_time_per_step,
            format_sep,
            format_added_values
        )
)