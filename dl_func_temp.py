import numpy

eps=1/100
def lrelu(x):
    if x<0:
        return x*eps
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
    activations={"A0":X}
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
        gradients["dW"+str(c)]=1/m*np.sum(dZ,axis=1,keepdims=True)
        if c>1:
            #descente de gradient = truc compliqué
            dZ = np.dot(parametres['W'+str(c)].T,dZ)*activations['A'+str(c-1)]*(1-activations['A'+str(c-1)])
    return gradients
    
def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2
    for c in range(1, C + 1):
        #nouveaux poids = ancien poids - learning rate x descente de grad poids
        parametres['W'+str(c)]=parametres['W'+str(c)]-learning_rate*gradients['dW'+str(c)]
        #nouveaux biais = ancien biais - learning rate x descente de grad biais
        parametres['b'+str(c)]=parametres['b'+str(c)]-learning_rate*gradients['db'+str(c)]
    return parametres

def predict(X, parametres):
    #tester le réseau
    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2
    Af = activations['A' + str(C)]
    return Af >= 0.5


def accuracy(TP, TN, FP, FN):
    return (TP+TN)/(TP+TN+FP+FN)

def precision(TP, FP):
    return TP/(TP+FP)

print(parametres)
