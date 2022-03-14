# -*- coding: utf-8 -*-

import os
import time
from json import loads as str2json
from json.decoder import JSONDecodeError
import tensorflow_hub as hub
import numpy as np
from falcon import HTTP_200

import simplelogging

log = simplelogging.get_logger()


def load_tf_model(cachep=True):
    "Load and cache(?) Tensorflow model"
    if cachep:
        cache_dir = os.path.join(os.getcwd(), "tf_models")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TFHUB_CACHE_DIR"] = cache_dir
        log.debug(f"Saving module to: {cache_dir}")

    model = hub.load(module_url)
    log.debug(f"Module loaded: {module_url}")
    return model


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"


class TFSentenceEncoder:
    def __init__(self, tf_model):
        # Parts shamelessly stolen from https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
        # Load the Tensorflow universal encoder model
        self.model = tf_model

    def run(self, input):
        """Embed 'input'"""
        log.info(f"(Worker: {os.getpid()})  Embedding {len(input)} messages")
        tic = time.perf_counter()
        retval = self.model(input)
        toc = time.perf_counter()
        log.info(f"(Worker: {os.getpid()})    \- took {toc - tic:0.4f} seconds")
        return retval.numpy().tolist()

    def on_post(self, req, resp):
        message = req.media.get("message")
        # Convert to a list if this is a JSON array
        try:
            messages = str2json(message)
        except JSONDecodeError:
            messages = [message]
        # Embed messages
        message_embeddings = self.run(messages)

        resp.media = {"embeddings": message_embeddings}
        resp.status = HTTP_200


class TFSentenceEncoderBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **_ignored):
        # FIXME: Put things like cache dir and model version as arguments here at some point
        if not self._instance:
            tf_model = load_tf_model()
            self._instance = TFSentenceEncoder(tf_model)
        return self._instance
