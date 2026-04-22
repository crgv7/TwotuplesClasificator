import numpy as np

def interception(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, ydiff)
    return div!=0 # no ocurre intercepcion


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def membership_triangle_function(value,line):
    m=(line[1][1]-line[0][1])/(line[1][0]-line[0][0])
    if value >=line[0][0] and value<=line[1][0] or value >=line[1][0] and value <=line[0][0]:
        if m>0:
            return (value - line[0][0])/(line[1][0]-line[0][0])
        else:
            return (line[1][0]-value)/(line[1][0]-line[0][0])
    else:
        return 0

def generate_fuzzy_set(n):
    set_fuzz={}
    interval=1/(n-1)
    x=0
    for i in range(0,n-1):
        if i ==0:
            line=[(x,1),(x+interval,0)]
            set_fuzz[i]=[line]
        else:
            line1=[(x-interval,0),(x,1)]
            line2=[(x,1),(x+interval,0)]
            set_fuzz[i]=[line1,line2]
        x+=interval

    line=[(x-interval,0),(x,1)]
    set_fuzz[n-1]=[line]

    return set_fuzz

def transform_to_fuzzy_set(value,linguistic_set):
    fuzzy_set=[]
    for set_point in linguistic_set.values():
        pair2=pair1=None
        enc=False
        for i in range(len(set_point)):
            for j in range(len(value)):
                if interception(set_point[i],value[j]):
                    x,y=line_intersection(set_point[i],value[j])
                    if y>0 and y<=1:
                        pair1=set_point[i]
                        pair2=value[j]
                        enc=True
        if enc:
            x,_=line_intersection(pair1,pair2)
            fuzzy_set.append(membership_triangle_function(x,pair1))
        else:
            fuzzy_set.append(0)

    return fuzzy_set

def fuzzy_set_2_tuple(fuzzy_set):
    a=np.sum(np.asarray([i*x for i,x in enumerate(fuzzy_set)]))
    b=np.sum(np.asarray([x for x in fuzzy_set]))
    beta=a/b
    index=np.round(beta)
    alpha=beta-index
    return (index,alpha)

def media_aritmetica(tuple_list):
    num=np.sum(np.asarray([tup[0]+tup[1] for tup in tuple_list]))
    beta=num/len(tuple_list)
    index=np.round(beta)
    alpha=beta-index
    return (index,alpha)
