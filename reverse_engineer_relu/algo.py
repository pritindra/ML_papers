# first layer
# network N, weight matrix W, bias vector b
# hyperplane Bz for each z, W.x + b = 0
# tensor model : need to add the algo there
p1 = []
p2 = []
s1 = []

def PointsOnLine(l):
    pass

def InferHyperplane(p):
    pass

def TestHyperplane(H):
    pass

def GetParams(H):
    pass

def algo():
    for t in range(1,L):
        # sample line segment l
        p1 = p1 + PointsOnline(l)
    
    for p in p1:
        H = InferHyperplane(p)
        if TestHyperplane(H):
            s1 = s2 + GetParams(H)
        else:
            p2 = p2 + p
    return p2, s1

    
