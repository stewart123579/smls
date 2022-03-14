# -*- coding: utf-8 -*-
"""My Machine Learning Server (smls)

A simple server to expose various endpoints for machine learning frameworks

Run by:
    gunicorn server:app
"""

import falcon
from smls import ml_services

import simplelogging

log = simplelogging.get_logger()
# log.setLevel(simplelogging.INFO)
# log.setLevel(simplelogging.DEBUG)


app = falcon.App()
app.add_route("/embed", ml_services.get("TFSentenceEncoder"))
