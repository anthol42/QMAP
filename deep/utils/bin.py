from pyutils import logger, Colors

# Available loggers: log, warn, error

log = logger.Logger(T=logger.LoggerType.INFO, name="INFO", logColor=None)
warn = logger.Logger(T=logger.LoggerType.WARNING, name="WARNING", logColor=None)
error = logger.Logger(T=logger.LoggerType.ERROR, name="ERROR", logColor=Colors.error)

def config_loggers_with_verbose(verbose: int = 3):
    log.config(show_name=True, show_time=True, show_type=False, show_origin=False,
               name_formatter=f"<CAPS>{Colors.success}[{'{}'}]{Colors.reset}",
               time_formatter=lambda x: (f'{Colors.darken}[{x.strftime("%Y-%m-%d %H:%M:%S")}]{Colors.reset}', True),
               start_sep=" ", end_sep="   ", console=verbose >= 2)

    warn.config(show_name=True, show_time=True, show_type=False, show_origin=True,
                name_formatter=f"<CAPS>{Colors.warning}[{'{}'}]{Colors.reset}",
                time_formatter=lambda x: (f'{Colors.darken}{x.strftime("%Y-%m-%d %H:%M:%S")}{Colors.reset}', False),
                start_sep="\n\t", origin_formatter=f"{Colors.darken} |  From: {'{}'} -- line no {'{}'}{Colors.reset}",
                console=verbose >= 1)

    error.config(show_name=True, show_time=True, show_type=False, show_origin=True,
                 name_formatter=f"<CAPS>{Colors.error}[{'{}'}]{Colors.reset}",
                 time_formatter=lambda x: (f'{Colors.darken}{x.strftime("%Y-%m-%d %H:%M:%S")}{Colors.reset}', False),
                 start_sep="\n\t", origin_formatter=f"{Colors.darken} |  From: {'{}'} -- line no {'{}'}{Colors.reset}",
                 console=verbose >= 0)

config_loggers_with_verbose()