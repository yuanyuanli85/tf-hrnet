import configparser
import ast


def load_net_cfg_from_file(cfgfile):
    def load_from_options(section, cfg):
        options = dict()
        xdict = dict(cfg.items(section))
        for key, value in xdict.items():
            try:
                value = ast.literal_eval(value)
            except:
                value = value
            options[key] = value
        return options

    cfg = configparser.ConfigParser()
    cfg.read(cfgfile)

    sections = cfg.sections()
    options = dict()
    for _section in sections:
        options[_section] = load_from_options(_section, cfg)

    return options
