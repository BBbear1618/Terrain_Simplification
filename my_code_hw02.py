#-- my_code_hw02.py
#-- hw02 GEO1015/2018
#-- [YOUR NAME] Yifang Zhao
#-- [YOUR STUDENT NUMBER] 4798899
#-- [YOUR NAME] Jinglan Li
#-- [YOUR STUDENT NUMBER] 4781937


import scipy.spatial
import numpy as np
import rasterio
import time


def read_pts_from_grid(jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that reads a grid in .tif format and retrieves the pixels as a list of (x,y,z) points shifted to the origin
     
    Input from jparams:
        input-file:  a string containing the path to a grid file with elevations
    Returns:
        a numpy array where each row is one (x,y,z) point. Each pixel from the grid gives one point (except for no-data pixels, these should be skipped).
    """
    print("=== Reading points from grid ===")

    # Tip: the most efficient implementation of this function does not use any loops. Use numpy functions instead.
    dataset = rasterio.open(jparams['input-file'])
    values = dataset.read(1)
    bb = dataset.bounds
    x_cellsize = dataset.transform[0]
    y_cellsize = dataset.transform[4]
    min_x = bb[0]
    min_y = bb[1]
    pts_list = []
    for i in range(dataset.height):      #row
        for j in range(dataset.width):   #column
            if values[i, j] != dataset.nodata:
                coor = dataset.xy(i, j)
                x = coor[0] + x_cellsize / 2  #center of the pixel
                y = coor[1] + y_cellsize / 2
                x-=min_x
                y-=min_y
                pts_list.append([x, y, values[i, j]])
    pts = np.array(pts_list)
    return pts


def simplify_by_refinement(pts, jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that takes a list of points and constructs a TIN with as few points as possible, while still satisfying the error-threshold. 

    This should be an implemented as a TIN refinement algorithm using greedy insertion. As importance measure the vertical error should be used.
     
    Input:
        pts:                a numpy array with on each row one (x,y,z) point
        from jparams:
            error-threshold:    a float specifying the maximum allowable vertical error in the TIN
    Returns:
        a numpy array that is a subset of pts and contains the most important points with respect to the error-threshold
    """
    print("=== TIN simplification ===")

    # Remember: the vertices of the initial TIN should not be returned
    time_start = time.time()
    #bbox
    min_x = min(pts[:,0])
    min_y = min(pts[:,1])
    max_x = max(pts[:,0])
    max_y = max(pts[:,1])
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    #enlarge 3X
    min_x -= delta_x
    max_x += delta_x
    min_y -= delta_y
    max_y += delta_y
    avg_z = np.mean(pts[:,2])
    lb = [min_x, min_y, avg_z]
    lu = [min_x, max_y, avg_z]
    rb = [max_x, min_y, avg_z]
    ru = [max_x, max_y, avg_z]
    
    pts_init = np.array([lb, lu, rb, ru])
    e = 1000.0
    e_max = jparams['error-threshold']
    while e > e_max:
        e = 0
        idx = 0
        tri = scipy.spatial.Delaunay(pts_init[:,:2])
        for i in range(len(pts)):
##            A = np.mat(pts_init[tri.simplices[tri.find_simplex(pts[i][0:2])]])
##            co = A.I * np.mat([1, 1, 1]).T
##            height = (1 - co[0, 0]*pts[i][0] - co[1, 0]*pts[i][1]) / co[2, 0] #interpolated height
            vtxs = pts_init[tri.simplices[tri.find_simplex(pts[i][0:2])]]
            x = vtxs[:,0]
            y = vtxs[:,1]
            z = vtxs[:,2]
            w1 = abs((1/2)*(pts[i][0]*y[1]+x[1]*y[2]+x[2]*pts[i][1]-pts[i][0]*y[2]-x[1]*pts[i][1]-x[2]*y[1]))
            w2 = abs((1/2)*(pts[i][0]*y[2]+x[2]*y[0]+x[0]*pts[i][1]-pts[i][0]*y[0]-x[2]*pts[i][1]-x[0]*y[2]))
            w3 = abs((1/2)*(pts[i][0]*y[0]+x[0]*y[1]+x[1]*pts[i][1]-pts[i][0]*y[1]-x[0]*pts[i][1]-x[1]*y[0]))
            height = (z[0]*w1 + z[1]*w2 + z[2]*w3)/(w1 + w2 + w3)
            dis = abs(height - pts[i][2])  #vertical error
            if  dis > e:
                e = dis
                q = pts[i]
                idx = i
        if e > e_max:
            pts_init = np.row_stack((pts_init, q))
            pts = np.delete(pts, idx, 0)
        
    pts_init = np.delete(pts_init, 0, 0)
    pts_init = np.delete(pts_init, 0, 0)
    pts_init = np.delete(pts_init, 0, 0)
    pts_init = np.delete(pts_init, 0, 0)
    print("Number of important points: {}".format(len(pts_init)))
    time_end = time.time()
    print("Totally cost:", time_end - time_start)
    return pts_init

    
def compute_differences(pts_important, jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that computes the elevation differences between the input grid and the Delaunay triangulation that is constructed from pts_important. The differences are computed for each pixel of the input grid by subtracting the grid elevation from the TIN elevation. The output is a new grid that stores these differences as float32 and has the same width, height, transform, crs and nodata value as the input grid.

    Input:
        pts_important:          numpy array with the vertices of the simplified TIN
        from jparams:
            input-file:                 string that specifies the input grid
            output-file-differences:    string that specifies where to write the output grid file with the differences
    """
    print("=== Computing differences ===")
    
    # original grid
    dataset = rasterio.open(jparams['input-file'])
    values = dataset.read(1)
    bb = dataset.bounds
    x_cellsize = dataset.transform[0]
    y_cellsize = dataset.transform[4]
    min_x = bb[0]
    min_y = bb[1]
    kwds = dataset.profile
    kwds['dtype'] = 'float32'
    
    # simplified TIN
    tri = scipy.spatial.Delaunay(pts_important[:,:2])

    
    #calculate differences
    new_values_list = []
    for i in range(dataset.height):      #row
        for j in range(dataset.width):   #column
            if values[i, j] != dataset.nodata:
                coor = dataset.xy(i, j)
                px = coor[0] + x_cellsize / 2
                py = coor[1] + y_cellsize / 2
                px-=min_x
                py-=min_y
                p = [px, py]
                idx = tri.find_simplex(p)
                if idx != -1:
                    vtxs = pts_important[tri.simplices[tri.find_simplex(p)]]
                    x = vtxs[:,0]
                    y = vtxs[:,1]
                    z = vtxs[:,2]
                    w1 = abs((1/2)*(p[0]*y[1]+x[1]*y[2]+x[2]*p[1]-p[0]*y[2]-x[1]*p[1]-x[2]*y[1]))
                    w2 = abs((1/2)*(p[0]*y[2]+x[2]*y[0]+x[0]*p[1]-p[0]*y[0]-x[2]*p[1]-x[0]*y[2]))
                    w3 = abs((1/2)*(p[0]*y[0]+x[0]*y[1]+x[1]*p[1]-p[0]*y[1]-x[0]*p[1]-x[1]*y[0]))
                    height = (z[0]*w1 + z[1]*w2 + z[2]*w3)/(w1 + w2 + w3)
                    difference = height - values[i, j]
                    new_values_list.append(difference)
                else:
                    # for points not inside the simplified TIN, nodata will be given
                    new_values_list.append(dataset.nodata)
                    
            else:
                new_values_list.append(dataset.nodata)
    new_values = np.array(new_values_list, dtype = 'float32')
    new_values = np.reshape(new_values, (dataset.height, dataset.width))
        
    # create new dataset and write
    with rasterio.open(jparams['output-file-differences'], 'w',
                                **kwds) as new_dataset:
        new_dataset.write(new_values, 1)

if __name__ == "__main__":
    jparams = dict({'input-file': 'el_capitan.tif', 'error-threshold': 10.0, 'output-file-tin': 'tin_100m.obj', 'output-file-differences': 'diff_100m.tif'})
    pts = read_pts_from_grid(jparams)
    pts_important = simplify_by_refinement(pts, jparams)
