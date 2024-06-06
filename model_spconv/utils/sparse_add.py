def sparse_add(t1, t2):
    return t1.replace_feature(t1.features + t2.features)