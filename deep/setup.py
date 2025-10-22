# The following file doesn't execute any task, except configuring the tools.
from pyutils import Colors, RGBColor, TraceBackColor, progress, Color
from pyutils.progress import format_total, format_dl_eta, format_time_per_step, format_sep, format_added_values
import sys, os

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