from .object_factory import ObjectFactory


class MLServiceProvider(ObjectFactory):
    """Class for providing ML services"""

    def get(self, service_id, **kwargs):
        return self.create(service_id, **kwargs)
