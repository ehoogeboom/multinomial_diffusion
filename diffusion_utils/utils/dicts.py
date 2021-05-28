import copy

def clean_dict(d, keys):
    d2 = copy.deepcopy(d)
    for key in keys:
        if key in d2:
            del d2[key]
    return d2
