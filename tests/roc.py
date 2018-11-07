from . import iter_module

modules = list(iter_module('roc'))

y_true = [1, 1, 1, 0]
y_pred = [0.9, 1, 0, 0]

for m in modules:
    results = m.roc_curve(y_true, y_pred)
    print(m, results)
