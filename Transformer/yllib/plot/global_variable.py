def _init(): 
    global _global_dict
    _global_dict = {}

def set_value(key, value):
    '''define global variable'''
    _global_dict[key] = value

def get_value(key, default=None):
    '''get global variable'''
    try:
        return _global_dict[key]
    except:
        return  default
