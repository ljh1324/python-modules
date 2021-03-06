import numpy as np

"""
Examples

Using array-scalar type:

>>> np.dtype(np.int16)
dtype('int16')
Structured type, one field name ‘f1’, containing int16:

>>> np.dtype([('f1', np.int16)])
dtype([('f1', '<i2')])
Structured type, one field named ‘f1’, in itself containing a structured type with one field:

>>> np.dtype([('f1', [('f1', np.int16)])])
dtype([('f1', [('f1', '<i2')])])
Structured type, two fields: the first field contains an unsigned int, the second an int32:

>>> np.dtype([('f1', np.uint), ('f2', np.int32)])
dtype([('f1', '<u4'), ('f2', '<i4')])
Using array-protocol type strings:

>>> np.dtype([('a','f8'),('b','S10')])
dtype([('a', '<f8'), ('b', '|S10')])
Using comma-separated field formats. The shape is (2,3):

>>> np.dtype("i4, (2,3)f8")
dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])
Using tuples. int is a fixed type, 3 the field’s shape. void is a flexible type, here of size 10:

>>> np.dtype([('hello',(np.int,3)),('world',np.void,10)])
dtype([('hello', '<i4', 3), ('world', '|V10')])
Subdivide int16 into 2 int8‘s, called x and y. 0 and 1 are the offsets in bytes:

>>> np.dtype((np.int16, {'x':(np.int8,0), 'y':(np.int8,1)}))
dtype(('<i2', [('x', '|i1'), ('y', '|i1')]))
Using dictionaries. Two fields named ‘gender’ and ‘age’:

>>> np.dtype({'names':['gender','age'], 'formats':['S1',np.uint8]})
dtype([('gender', '|S1'), ('age', '|u1')])
Offsets in bytes, here 0 and 25:

>>> np.dtype({'surname':('S25',0),'age':(np.uint8,25)})
dtype([('surname', '|S25'), ('age', '|u1')])

"""
a = np.full((1024, 1024, 11), 1)
b = np.zeros((1024, 1024))
row, col, filter = a.shape

print(b.shape)

wantfilter = 0
for i in range(row):
    for j in range(col):
        b[i][j] = a[i][j][wantfilter]
print(b)
print(b.shape)

b = a[:][:][wantfilter]
print(b.shape)
