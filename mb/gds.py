import gdsCAD as gc
import numpy as np


def recenter(arr,bounding_box=None):
    if bounding_box is None:
        b = arr.bounding_box
    else:
        b=bounding_box
    dx = (b[0,0]+b[1,0])/2
    dy = (b[0,1]+b[1,1])/2
    #print(dx,dy,int64(bounding_box.ravel()))
    arr.origin=np.array(arr.origin)+0.0 
    arr.translate([-dx,-dy])


def sign_cell(cell,name,size = 16, sep=16,center=True,layer=1):
    bb = cell.bounding_box
    label = gc.shapes.LineLabel('', size)
    label.add_text( name ,  'romand')
    label.layer=layer
    bl = label.bounding_box
    if center:
        width = (bl[1,0]-bl[0,0])
    else:
        width=0
    label.translate( ((bb[0,0]+bb[1,0])/2-width/2 , bb[1,1]+sep)  )
    cell.add(label)


def pack(cells,mode='vertical',margin=50,name='pack'):
    """ pack / assemble next to each other cells in list 'cells'

    mode 'vertical' or 'horizontal'
    """
    tar = gc.core.Cell(name)
    tar.add(cells[0])
    bb = cells[0].bounding_box
    #width  = bb[1,0]-bb[0,0]
    #height = -bb[0,1]+bb[1,1]
    width  = -(bb[0,0]-bb[1,0])
    height = -(bb[0,1]-bb[1,1]    )
    
    for c in cells[1:]:
        print(c)
        bb2=c.bounding_box
        #ref2.origin=(bb[0,0]-bb2[0,0],bb[0,1]-bb[0,1]+height)
        ref2 = gc.core.CellReference(c) # reference to cell
        if mode=='vertical':
            ref2.origin=(0,-bb2[0,1]+height+margin)
        elif mode=='horizontal':
            ref2.origin=(-bb2[0,0]+width+margin,0)
        else:
            ref2.origin=(0,-bb2[0,1]+height+margin)

        #ref2.origin=(0,height+margin)#(bb[0,1]-bb2[1,1],bb[0,0]+bb2[0,0])
        tar.add(ref2)
        print(height)
        bb=tar.bounding_box
        width  = -(bb[0,0]-bb[1,0])
        height = -(bb[0,1]-bb[1,1]    )
        
    return tar
    
    

def pack2(cells,mode='vertical',margin=50,name='pack',dgrid=100):
    """ pack / assemble next to each other cells in list 'cells'

    mode 'vertical' or 'horizontal'
    """
    tar = gc.core.Cell(name)
    tar.add(cells[0])
    print(cells[0])
    bb = cells[0].bounding_box
    #width  = bb[1,0]-bb[0,0]
    #height = -bb[0,1]+bb[1,1]
    width  = -(bb[0,0]-bb[1,0])
    height = -(bb[0,1]-bb[1,1]    )

    width = np.ceil(width/dgrid)*dgrid
    height = np.ceil(height/dgrid)*dgrid
    
    for c in cells[1:]:
        #print(c)
        bb2=c.bounding_box
        #ref2.origin=(bb[0,0]-bb2[0,0],bb[0,1]-bb[0,1]+height)
        ref2 = gc.core.CellReference(c) # reference to cell
        if mode=='vertical':
            ref2.origin=(0,-bb2[0,1]+height+margin)
        elif mode=='horizontal':
            ref2.origin=(-bb2[0,0]+width+margin,0)
        else:
            ref2.origin=(0,-bb2[0,1]+height+margin)

        #ref2.origin=(0,height+margin)#(bb[0,1]-bb2[1,1],bb[0,0]+bb2[0,0])
        tar.add(ref2)
        #print(height)
        bb=tar.bounding_box
        width  = -(bb[0,0]-bb[1,0])
        height = -(bb[0,1]-bb[1,1]    )
        width = np.ceil(width/dgrid)*dgrid
        height = np.ceil(height/dgrid)*dgrid
        
    return tar
    
    
def line_grid2(pitch,le,n=None):
    if n is None:
        n= int(le/(2*pitch))
    di = gc.shapes.Rectangle((0,0),(pitch/2,le))
    grid  = gc.core.CellArray(di, n, 1, [pitch,1] , origin=(0,0))
    grid_cell = core.Cell('grid')
    grid_cell.add(grid)
    #grid_cell.show()
    return grid_cell

def line_grid(a,pitch=None,le=200,n=None):
    if pitch is None:
        pitch = 2*a
    if n is None:
        n= int(le/(2*pitch))
    di = gc.shapes.Rectangle((0,0),(a,le))
    cell = core.Cell('pi')
    cell.add(di)
    
    grid  = gc.core.CellArray(cell, n, 1, [pitch,1] , origin=(0,0))
    grid_cell = core.Cell('grid')
    grid_cell.add(grid)
    #grid_cell.show()
    return grid_cell

def line_grid_chirp(a,pitch1,pitch2,le=200,n=None):
    if n is None:
        n= int(le/(2*pitch1))
    di = gc.shapes.Rectangle((0,0),(a,le))
    cell = core.Cell('pi')
    cell.add(di)

    #di.len=lambda : 1
    grid_cell = core.Cell('grid')
    
    dx = np.linspace(pitch1,pitch2,n)
    x = dx.cumsum()
    for i in range(len(x)):
        ref2 = core.CellReference(cell) # reference to cel
        ref2.origin=(x[i],0)        
        grid_cell.add(ref2)
        del ref2
        
    return grid_cell

def cavity(di,points):
    grid_cell = core.Cell('cavity')    
    for i in range(len(points)):
        ref2 = core.CellReference(di) # reference to cel
        ref2.origin=(points[i,0],points[i,1])        
        grid_cell.add(ref2)
        #del ref2
        
    return grid_cell
    



def hex_array_points(a,n):
    """ returns points for hex grid

    cell2 = core.Cell('HEX')
    hex   = shapes.RegPolygon((0,0), 4, 6)
    hex.rotate(30)
    cell2.add(hex)
    p=hex_array_points(10,30)
    
    for i in range(len(p)):
      ref2 = core.CellReference(cell2) # reference to cel
      ref2.magnification=1
      ref2.origin=(p[i,0],p[i,1])
      top.add(ref2)
      del ref2
    # also:
    # arr3 = core.CellArray(cell, 3, 5, ((a*sqrt(3)/2, a/2), (a*sqrt(3)/2, -a/2)), origin = (300, 50))
    """
    h = 2*a*np.sin(np.pi/3)
    p = [[-a, 0],
         #[-a, h],
         [0,0],
         [-a/2,h/2],
         [a/2,h/2],
         [a,0],
         [3/2*a,h/2],
         ]    
    p=np.array(p)
    hx=[]
    for kk in range(n//2):
        for k in range(n):
            xx = p[:,0]+kk*3*a
            yy = p[:,1]+k*1*h
            hx.append([xx,yy])         
    hh = np.hstack(hx).T
    return hh

def make_rect_array(di,pitch,n=10):
    sing = gc.core.Cell(str(di).split()[0]) # new cell name by shape 
    sing.add(di)
    rect_arrays = gc.core.Cell('array_'+str(di).split()[0])
    arr  = gc.core.CellArray(sing, n, n, [pitch]*2 , origin=(0,0))

    b = arr.bounding_box
    dx = (b[0,0]+b[1,0])/2
    dy = (b[0,1]+b[1,1])/2
    arr.origin=arr.origin+0.0 
    arr.translate([-dx,-dy])

    rect_arrays.add(arr)

    return rect_arrays

    

def make_hex_array(di,pitch,n=10):
    # new cell to place shape:
    sing = gc.core.Cell(str(di).split()[0]) # new cell name by shape
    # palce shape there
    sing.add(di)
    # new cell for an array
    cell = gc.core.Cell('array_Hex_'+str(di).split()[0])

    p = hex_array_points(pitch,n)
    for i in range(len(p)):
        ref2 = core.CellReference(sing) # reference to cel
        ref2.magnification=1
        ref2.origin=(p[i,0],p[i,1])
        cell.add(ref2)
        del ref2

    bb=cell.bounding_box
    for c in cell.elements:
        recenter(c,bb)
        
    return cell


def make_chirped_array(di,pitch0,pitch1,n=20):
    # new cell to place shape:
    sing = gc.core.Cell(str(di).split()[0]) # new cell name by shape
    # palce shape there
    sing.add(di)
    # new cell for an array
    cell = gc.core.Cell('array_Chirp_'+str(di).split()[0])

    pitchM = (pitch1+pitch0)/2
    y  = np.linspace(0,n*pitchM,n)
    dx = np.linspace(pitch0,pitch1,n)
    x = dx.cumsum()
    xx,yy = np.meshgrid(x,y)
    p= np.array([xx.ravel(),yy.ravel()]).T
    for i in range(len(p)):
        ref2 = core.CellReference(sing) # reference to cel
        ref2.origin=(p[i,0],p[i,1])        
        cell.add(ref2)
        del ref2

    # center the CELL
    bb=cell.bounding_box # cell Bbox
    for c in cell.elements:
        recenter(c,bb) # center each element in d cell
        
    return cell

def make_chirped_array2(di,pitch0,pitch1,n=20):
    # new cell to place shape:
    sing = gc.core.Cell(str(di).split()[0]) # new cell name by shape
    # palce shape there
    sing.add(di)
    # new cell for an array
    cell = gc.core.Cell('array_Chirp_'+str(di).split()[0])

    pitchM = (pitch1+pitch0)/2
    y  = np.linspace(0,n*pitchM,n)
    dx = np.linspace(pitch0,pitch1,n)
    x = dx.cumsum()
    xx,yy = np.meshgrid(x,x)
    p= np.array([xx.ravel(),yy.ravel()]).T
    for i in range(len(p)):
        ref2 = core.CellReference(sing) # reference to cel
        ref2.origin=(p[i,0],p[i,1])        
        cell.add(ref2)
        del ref2

    # center the CELL
    bb=cell.bounding_box # cell Bbox
    for c in cell.elements:
        recenter(c,bb) # center each element in d cell
        
    return cell



###############################################################################
###############################################################################
# fix bug in boolean operation in gdsCAD 
###############################################################################

import numpy as np
import gdspy
from gdsCAD import *
import gdsCAD as gc
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

# to avoid np.ndarray errors for gdspy polygons converted from gdsCAD, extract  the polygon points and reconstruct the polygon
def pyply(ply, ly):
    return gdspy.Polygon([list(i) for i in ply], layer=ly)
#end
# function to turn gdspy polygons into gdsCAD Boundary object
def pypol2bdy(plst, ly):
    return core.Boundary(plst.polygons, layer=ly)
#end



# the following definition can take in any boundary object
def boolean(obj01, obj02, operation, ly):
    el_cad = core.Elements()
    cl_aa = gc.core.Cell('AA')
    cl_aa.add(obj02)
    cl_mm = gc.core.Cell('MM')
    cl_mm.add(obj01)
    el_aa = gc.core.Elements(cl_aa.flatten(), layer=ly)
    el_mm = gc.core.Elements(cl_mm.flatten(), layer=ly)    #, obj_type='boundaries'
    aaa = gdspy.PolygonSet([ii.points for ii in el_aa], layer=ly)
    mmm = gdspy.PolygonSet([ii.points for ii in el_mm], layer=ly)
    aaa_mmm = gdspy.boolean([pyply(ii,1) for ii in mmm.polygons] , [pyply(ii,1) for ii in aaa.polygons], operation)
    #k=0
    for ii in aaa_mmm.polygons:
        #el_cad.add(gc.core.Boundary(map(tuple, ii), layer=ly))
        el_cad.add(gc.core.Boundary(ii, layer=ly)) # removed this mapp error ...but dont know why.........
    return el_cad


def subtract(obj01, obj02, layer=None):
    if layer is None:
        layer=obj01.layer
    return boolean(obj01, obj02, 'not', layer)


def add(obj01, obj02, layer=None):
    if layer is None:
        layer=obj01.layer
    return boolean(obj01, obj02, 'or', layer)


def intersect(obj01, obj02, layer=None):
    if layer is None:
        layer=obj01.layer
    return boolean(obj01, obj02, 'and', layer)

def xor(obj01, obj02, layer=None):
    if layer is None:
        layer=obj01.layer
    return boolean(obj01, obj02, 'xor', layer)

def negate(r1,margin=0):
    b=gc.shapes.Rectangle(r1.bounding_box[0,:],r1.bounding_box[1,:])
    r2 = subtract(b,r1)
    margin=np.array(margin)
    if np.prod(abs(margin))>0:
        b=gc.shapes.Rectangle(r1.bounding_box[0,:]+margin,r1.bounding_box[1,:]-margin)
        r2 = intersect(b,r2)
    cell = gc.core.Cell('negate')
    cell.add(r2)
    return cell


def gds_move_all_to_layer1(fname):# if too many elemenst inf slow....
    gds = gc.core.GdsImport(fname)
    
    def move2layer(cell,lay=1):
        for el in cell.elements:
            if type(el) is gc.core.CellReference :
                move2layer(el.ref_cell)
            else:
                el.layer=lay
    gds_cells=[]
    for key in list(gds):
        cell = gds[key]
        move2layer(cell,lay=1)
        gds_cells.append(cell)
        
    layout = core.Layout('LIBRARY')
    [layout.add(c) for c in gds_cells] 
    layout.save(fname[:-4]+"one_layer.gds")
        
############# https://heitzmann.github.io/gdstk/gettingstarted.html#first-layout


def gdstk_gds_move_all_to_layer1(fname): # OK!
    """move everything to layer 1"""
    import gdstk

    gds=gdstk.read_gds(fname)
    print("read")
    
    cells = gds.top_level()
    def move2layer(cell,lay=1):
        for i in range(len(cell.polygons)):
            cell.polygons[i].layer = lay            
        for i in range(len(cell.paths)):
            cell.paths[i].set_layers(1)
            
        for i in range(len(cell.references)):
            move2layer(cell.references[i].cell)

    for c in cells:
        move2layer(c)
            #ww.references[0].cell.references[1].cell.polygons[0].layer
    gds.write_gds(fname[:-4]+"_gstk_one_layer1.gds")


def gdstk_gds_move_layers(fname, layer_in, layer_out): 
    """move stuff from layer "layer_in" to "layer_out" """
    import gdstk

    gds = gdstk.read_gds(fname)
    print("read")
    
    cells = gds.top_level()
    def move2layer(cell,layer_in,layer_out):
        
        for i in range(len(cell.polygons)):
            if cell.polygons[i].layer == layer_in:
                cell.polygons[i].layer = layer_out   

        for i in range(len(cell.paths)): 
            if cell.paths[i].layers[0]== layer_in:
                cell.paths[i].set_layers(layer_out)
            
        for i in range(len(cell.references)):
            move2layer(cell.references[i].cell,layer_in,layer_out)

    for c in cells:
        move2layer(c)
            #ww.references[0].cell.references[1].cell.polygons[0].layer
    gds.write_gds(fname[:-4]+"_gstk_one_layer1.gds")
    
