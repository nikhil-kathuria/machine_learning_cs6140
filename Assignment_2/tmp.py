from NN import sigmoidAll
import numpy as np

Wjk = np.loadtxt('Wjk', dtype=float)

Hidden = np.loadtxt('Hid', dtype=float)

np.set_printoptions(suppress=True)

print( sigmoidAll(Hidden.dot(np.transpose(Wjk)) ))

print()

print( np.transpose(sigmoidAll(Wjk.dot(np.transpose(Hidden)))) )

