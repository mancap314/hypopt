import hypopt

methods, X, y = [], [], []

def set_methods(list):
    global methods
    methods = list

def get_methods():
    global methods
    return methods

def set_X(vars):
    global X
    X = vars

def get_X():
    global X
    return X

def set_y(var):
    global y
    y = var

def get_y():
    global y
    return y

def runs():
    for method in methods:
        hypopt.run_method(X, y, method)

