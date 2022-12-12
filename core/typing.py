import sys
import copy
import collections
from jax import tree_util

from jax_tools import jax_utils


""" Attribute Dictionary """
@tree_util.register_pytree_node_class
class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def asdict(self, shallow=False):
        if shallow:
            return {k: v for k, v in self.items()}
        res = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                res[k] = v.asdict()
            else:
                res[k] = copy.deepcopy(v)

        return res
    
    def __getattr__(self, name):
        return self[name] if name in list(self) else None

    def __new__(cls, *args, **kwargs):
        d = super().__new__(cls, *args, **kwargs)
        return dict2AttrDict(d)
    
    def __getnewargs__(self):
        return (AttrDict.asdict(self),)

    """ The following three methods enable pickling """
    def __getstate__(self):
        return (self.asdict(),)
        
    def __setstate__(self, state):
        self.update(state[0])

    def __reduce__(self):
        return (AttrDict, (), self.__getstate__())

    def tree_flatten(self):
        children = tuple(self.values())
        aux_data = tuple(self.keys())
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        t = cls()
        for k, v in zip(aux_data, children):
            t[k] = v
        return t
    
    def copy(self):
        res = AttrDict()
        for k, v in self.items():
            res[k] = v
        return res
    
    def subdict(self, *args):
        res = AttrDict()
        for k in args:
            assert k in self, (k, list(self))
            res[k] = self[k]
        return res
    
    def slice(self, loc):
        return jax_utils.tree_map(lambda x: x[loc], self)


def subdict(d, *args):
    res = type(d)()
    for k in args:
        assert k in d, (k, list(d))
        res[k] = d[k]
    return res

def exclude_subdict(d, *args):
    res = type(d)()
    for k in d.keys():
        if k in args:
            continue
        res[k] = d[k]
    return res

def dict2AttrDict(d: dict, shallow=False, to_copy=False):
    if isinstance(d, AttrDict) and not to_copy:
        return d
    if shallow:
        res = AttrDict()
        for k, v in d.items():
            res[k] = v
        return res

    res = AttrDict()
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = dict2AttrDict(v, to_copy=to_copy)
        else:
            res[k] = copy.deepcopy(v)

    return res

def AttrDict2dict(attr_dict: AttrDict, shallow=False):
    if isinstance(attr_dict, AttrDict):
        return attr_dict.asdict(shallow=shallow)
    else:
        return attr_dict


""" NamedTuple with the support of array slice """
def namedarraytuple(typename, field_names, return_namedtuple_cls=False,
        classname_suffix=False):
    """
    Returns a new subclass of a namedtuple which exposes indexing / slicing
    reads and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).
    (Code follows pattern of collections.namedtuple.)
    >>> PointsCls = namedarraytuple('Points', ['x', 'y'])
    >>> p = PointsCls(np.array([0, 1]), y=np.array([10, 11]))
    >>> p
    Points(x=array([0, 1]), y=array([10, 11]))
    >>> p.x                         # fields accessible by name
    array([0, 1])
    >>> p[0]                        # get location across all fields
    Points(x=0, y=10)               # (location can be index or slice)
    >>> p.get(0)                    # regular tuple-indexing into field
    array([0, 1])
    >>> x, y = p                    # unpack like a regular tuple
    >>> x
    array([0, 1])
    >>> p[1] = 2                    # assign value to location of all fields
    >>> p
    Points(x=array([0, 2]), y=array([10, 2]))
    >>> p[1] = PointsCls(3, 30)     # assign to location field-by-field
    >>> p
    Points(x=array([0, 3]), y=array([10, 30]))
    >>> 'x' in p                    # check field name instead of object
    True
    """
    nt_typename = typename
    if classname_suffix:
        nt_typename += "_nt"  # Helpful to identify which style of tuple.
        typename += "_nat"

    try:  # For pickling, get location where this function was called.
        # NOTE: (pickling might not work for nested class definition.)
        module = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        module = None
    NtCls = collections.namedtuple(nt_typename, field_names, module=module)

    def __getitem__(self, loc):
        try:
            return type(self)(*({k: v[loc] for k, v in s.items()} 
                                if isinstance(s, dict) else s[loc] for s in self))
        except IndexError as e:
            for j, s in enumerate(self):
                if s is None:
                    continue
                try:
                    _ = s[loc]
                except IndexError:
                    raise Exception(f"Occured in {self.__class__} at field "
                        f"'{self._fields[j]}'.") from e

    __getitem__.__doc__ = (f"Return a new {typename} instance containing "
        "the selected index or slice from each field.")

    def __setitem__(self, loc, value):
        """
        If input value is the same named[array]tuple type, iterate through its
        fields and assign values into selected index or slice of corresponding
        field.  Else, assign whole of value to selected index or slice of
        all fields.  Ignore fields that are both None.
        """
        if not (isinstance(value, tuple) and  # Check for matching structure.
                getattr(value, "_fields", None) == self._fields):
            # Repeat value for each but respect any None.
            value = tuple({k: v[loc] for k, v in s.items()} 
                          if isinstance(s, dict) else value for s in self)
        try:
            for j, (s, v) in enumerate(zip(self, value)):
                if isinstance(s, dict):
                    for k in s.keys():
                        s[k][loc] = v[k]
                else:
                    s[loc] = v
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(f"Occured in {self.__class__} at field "
                f"'{self._fields[j]}'.") from e

    def __contains__(self, key):
        "Checks presence of field name (unlike tuple; like dict)."
        return key in self._fields

    def __getattr__(self, name):
        return self[name] if name in self._fields else None

    """ The following three methods enable pickling """
    def __getstate__(self):
        try:
            return [s for s in self]
        except IndexError as e:
            for j, s in enumerate(self):
                if s is None:
                    continue
                try:
                    _ = s
                except IndexError:
                    raise Exception(f"Occured in {self.__class__} at field "
                        f"'{self._fields[j]}'.") from e
        
    def __setstate__(self, state):
        value = state[0]
        try:
            for j, (s, v) in enumerate(zip(self, value)):
                if isinstance(s, dict):
                    for k in s.keys():
                        s[k] = v[k]
                else:
                    s = v
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(f"Occured in {self.__class__} at field "
                f"'{self._fields[j]}'.") from e

    def __reduce__(self):
        return (tuple, (), self.__getstate__())

    def get(self, key, val=None):
        "Retrieve value as if indexing into regular tuple."
        return getattr(self, key) if key in self._fields else val

    def get_with_idx(self, index):
        "Retrieve value as if indexing into regular tuple."
        return tuple.__getitem__(self, index)

    def items(self):
        "Iterate ordered (field_name, value) pairs (like OrderedDict)."
        for k, v in zip(self._fields, self):
            yield k, v

    def tuple_itemgetter(i):
        def _tuple_itemgetter(obj):
            return tuple.__getitem__(obj, i)
        return _tuple_itemgetter

    for method in (__getitem__, __setitem__, get, items):
        method.__qualname__ = f'{typename}.{method.__name__}'

    arg_list = repr(NtCls._fields).replace("'", "")[1:-1]
    class_namespace = {
        '__doc__': f'{typename}({arg_list})',
        '__slots__': (),
        '__getitem__': __getitem__,
        '__setitem__': __setitem__,
        '__getattr__': __getattr__,
        '__contains__': __contains__,
        '__getstate__': __getstate__, 
        '__setstate__': __setstate__, 
        'get': get,
        'get_with_idx': get_with_idx,
        'items': items,
    }

    RESERVED_NAMES = ("get", "items")
    for index, name in enumerate(NtCls._fields):
        if name in RESERVED_NAMES:
            raise ValueError(f"Disallowed field name: {name}.")
        itemgetter_object = tuple_itemgetter(index)
        doc = f'Alias for field number {index}'
        class_namespace[name] = property(itemgetter_object, doc=doc)

    result = type(typename, (NtCls,), class_namespace)
    result.__module__ = NtCls.__module__

    # register NtCls as pytree node
    tree_util.register_pytree_node(
        NtCls,
        lambda xs: (tuple(xs), None),  # tell JAX how to unpack to an iterable
        lambda _, xs: NtCls(*xs)       # tell JAX how to pack back into a Point
    )

    if return_namedtuple_cls:
        return result, NtCls
    return result


def dict2namedtuple(dict, typename):
    return namedarraytuple(typename, list(dict))(*dict.values())


def namedtuple2dict(nt):
    return nt._asdict()


def namedtuple2dict(nt, shallow=True):
    if shallow:
        return AttrDict(nt._asdict())
    return dict2AttrDict(nt._asdict())


"""
root_dir: logdir/env_name/algo_name
model_name: base_name/a{id}/i{iteration}-v{version}
"""
ModelPath = collections.namedtuple('ModelPath', 'root_dir model_name')


def construct_model_name_from_version(base, iteration, version):
    return f'{base}/i{iteration}-v{version}'

def construct_model_name(base, aid, iteration, version):
    return f'{base}/a{aid}/i{iteration}-v{version}'

def get_aid(model_name: str):
    _, aid, _ = model_name.rsplit('/', maxsplit=2)
    aid = eval(aid[1:])
    assert isinstance(aid, int), aid
    return aid

def get_vid(model_name: str):
    _, vid = model_name.rsplit('/', maxsplit=1)
    vid = vid.rsplit('v', maxsplit=1)[-1]
    return vid

def get_aid_vid(model_name: str):
    _, aid, vid = model_name.rsplit('/', maxsplit=2)
    aid = eval(aid[1:])
    vid = vid.rsplit('v', maxsplit=1)[-1]
    assert isinstance(aid, int), aid
    return aid, vid

def get_all_ids(model_name: str):
    _, aid, vid = model_name.rsplit('/', maxsplit=2)
    aid = eval(aid[1:])
    iid, vid = vid.split('v', maxsplit=1)
    iid = eval(iid[1:-1])
    assert isinstance(aid, int), aid
    assert isinstance(iid, int), iid
    return aid, iid, vid

def get_basic_model_name(model_name: str):
    """ Basic model name excludes aid and vid """
    name = '/'.join(model_name.split('/')[:3])

    return name

def get_algo(model: ModelPath):
    s = model.root_dir.split('/')
    algo = s[-1]
    # if len(s) == 1:
    #     algo = model.root_dir
    # elif len(s) == 3:
    #     algo = s[-1]
    # elif len(s) == 4:
    #     algo = ''.join(s[-2:])
    # else:
    #     # raise ValueError(f'Unknown model: {model}')
    #     assert False, f'Unknown model: {model}'

    return algo
