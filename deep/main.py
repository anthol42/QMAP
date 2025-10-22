import setup # Initiate project configurations
from datetime import datetime
import os
import typer
import pkgutil
import importlib
from utils.cli import Experiment

app = typer.Typer(pretty_exceptions_enable=False)

for module_info in pkgutil.iter_modules(['experiments']):
    module_name = f"experiments.{module_info.name}"
    module = importlib.import_module(module_name)
    for att in dir(module):
        obj = getattr(module, att)
        if isinstance(obj, Experiment):
            app.command(
                context_settings=dict(allow_extra_args=True, ignore_unknown_options=True),
            )(obj.fn)




if __name__ == "__main__":
    os.environ['TORCH_HOME'] = f'{os.getcwd()}/.cache'
    start = datetime.now()
    app()
    end = datetime.now()
    print(f"Done!  Total time: {(end - start)}")