class IdentifierConstructor:
    def __init__(self, default_name='default_identifier'):
        self._default_name = default_name
    
    @property
    def name(self):
        return self._default_name

    def get_identifier(self, *args, **kwargs):
        identifier = ''
        for v in args:
            if v is not None:
                identifier += f'_{v}' if identifier else f'{v}'
        for k, v in kwargs.items():
            if v is not None:
                v = f'{k}({v})'
                identifier += f'_{v}' if identifier else f'{v}'
        if identifier == '':
            identifier = self._default_name
        return identifier
