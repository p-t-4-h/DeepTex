import numpy

def lrelu(x):
    if x<0:
        return x*1/100
    else:
        return x

def initialisation(dimensions):
    parametres={}
    C=len(dimensions)
    for c in range(1,c):
        parametres["W"+str(c)]=np.random.randn(dimensions[c],dimensions[c-1])
        parametres["b"+str(c)]=np.random.randn(dimensions[c],1)

    return parametres

def forward_propagation(x, parametres):
    activations={"A0"=X}
    C=len(parametres) // 2
    for c in range(1,C+1):
        Z=parametres["W"+str(c)].dot(activations["A"+str(c-1)])+parametres["b"+str(c)]
        activations["A"+str(c)]=lrelu(Z)
    return activations

def back_propagation(y,activations,parametres):
    m = y.shape[1]
    C=len(parametres)//2
    dZ=activations["A"+str(C)]-y
    gradients={}

    for c in reversed(range(1,C+1)):
        gradients["dW"+str(c)]=1/m*np.dot(dZ, activations["A"+str(c-1)].T)