'''
Check if clockwise
https://www.tutorialspoint.com/Check-if-two-line-segments-intersect

polar angle = arctan( ... )
'''

import operator
import numpy as np

def vector_dist(p1,p2):
    return np.sqrt( (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def get_max(points, p):
    points.sort(key=lambda x: x[0])
    m_points = {} 
    for i in points:
        angle, vecotr = i
        if angle not in m_points.keys():
            m_points[angle] = [(vector_dist(p, vecotr), vecotr)]
        else:
            m_points[angle] += [(vector_dist(p, vecotr), vecotr)]

    to_return = [] 

    for k in m_points.keys():
        m_vector = max(m_points[k], key=lambda x: x[0])
        to_return.append( (k,m_vector[1]) ) 
    to_return.sort( key=lambda x:x[0])
    return to_return


def theta(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    h = vector_dist(p1,p2)
    a = abs( p1[0] - p2[0] )
    if a == 0:
        if p1[1] > p2[1]:
            #return (3.*np.pi)/2.
            return np.around( (3.*np.pi)/2., decimals=6)
        return np.around( np.pi/2, decimals=6)
    return np.around( np.arccos( a/h ), decimals=6) 
    #return np.arccos( a/h ), 


def clockwise( p1, p2, p3 ):
    v = (p2[1] - p1[1])*(p3[0]-p2[0]) - (p2[0]-p1[0])*(p3[1]-p2[1])
    return v


def graham_scan(raw_points:list):
    sorted_points = sorted(raw_points, key=operator.itemgetter(0,1) )
    p0 = sorted_points[0]
    points = [ (theta(p0,sorted_points[x]),sorted_points[x]) for x in range(1,len(sorted_points))] 
    points = get_max(points, p0)
    stack = [p0]
    for r in points:
        angle, p = r
        while( len(stack)>2 and clockwise( stack[-2], stack[-1], p) > 0 ):
            stack.pop()
        stack.append(p)
    return stack 


if __name__=='__main__':
    p1 = (0,0)
    p2 = (3,0)
    p3 = (3,3)
    p4 = (0,3)
    p5 = (1,1)
    p6 = (0,2)
    p7 = (4,5)

    p = [p1, p2, p3, p4, p5, p6, p7]
    print(graham_scan(p))











