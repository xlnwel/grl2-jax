from collections import defaultdict


class IdentifierConstructor:
    def __init__(self, default_name='default_identifier'):
        self._default_name = default_name
    
    @property
    def name(self):
        return self._default_name

    def get_identifier(self, *args):
        identifier = ''
        for v in args:
            if v is not None:
                identifier += f'_{v}' if identifier is not None else v
        if identifier == '':
            identifier = self._default_name
        return identifier
