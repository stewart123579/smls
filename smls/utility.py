"""Utility functions"""

import json

import simplelogging

log = simplelogging.get_logger()


def clean_dict(**kwargs):
    """Remove any 'None's from a dict"""
    clean = dict((k, v) for k, v in kwargs.items() if v is not None)
    log.debug(clean)
    return clean


def dict_to_string(it):
    """Convert a dict to a sting (keys are sorted)"""
    return json.dumps(it, sort_keys=True)
