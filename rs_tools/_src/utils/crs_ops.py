#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:49:15 2020

@author: ghiggi
"""
import numpy as np
import cartopy.crs as ccrs

def get_bbox_polygon(bbox): 
    from shapely.geometry.polygon import Polygon
    p = Polygon(((bbox[0],bbox[1]), # bottom left
                (bbox[0],bbox[3]),  # top left
                (bbox[2],bbox[3]),  # top rght
                (bbox[2],bbox[1]),  # bottom right
                (bbox[0],bbox[1]))) # bottom left
    return p

def get_bbox_coords(bbox, output_type='xy'):
    """Return corner coordinates of bounding box.
    # bbox : (xmin, ymin, xmax, ymax) 
    
    # output_type = 'shapely', 'array', 'xy', 'tuple'
    # shapely: shapely polygon 
    # array : numpy array with xy columns
    # xy : x, y numpy arrays 
    # tuple: list of (x,y)  
    
    # Return (bottom_left, top_left, top_right bottom_right)
    """
    if (output_type not in ['xy','tuple','array']):
        raise ValueError("Valid output_type: 'xy','tuple','array'")
    x = np.array([bbox[0], bbox[0], bbox[2], bbox[2]])
    y = np.array([bbox[1], bbox[3], bbox[3], bbox[1]])
    if (output_type=='xy'):
        return x, y
    if (output_type=='tuple'):
        return list(zip(x,y))
    if (output_type=='array'):
        return np.vstack((x,y)).transpose()
    

def get_bbox(x,y):
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    return [xmin, ymin, xmax, ymax]

def get_projected_bbox(bbox, crs_ref, crs_proj, lonlat=False): 
    # Check if xmin > xmax (crossing the antimeridian)
    # - This only occurs for geographic coordinates / projections
    # TODO : Infer if lonlat from CRS 
    flag_crossing = False 
    if (bbox[0] > bbox[2]):
        flag_crossing = True
    #------------------------------------------------------------.   
    x, y = get_bbox_coords(bbox, output_type="xy")  
    xy_prj = crs_proj.transform_points(src_crs=crs_ref, x=x, y=y)[:,0:2]
    bbox_proj = get_bbox(xy_prj[:,0], xy_prj[:,1]) # this simply look at min and max
    #------------------------------------------------------------.  
    # Deal with crossing antimeridian
    if (flag_crossing is True) and (lonlat is True): 
        xmax = bbox_proj[0]
        xmin = bbox_proj[2]
        bbox_proj[0] = xmin 
        bbox_proj[2] = xmax 
    #------------------------------------------------------------.      
    return bbox_proj


def adapt_bbox_for_resolution(bbox, resolution, lonlat=False):
    """ Adapt bbox to allow specified resolution. """
    # Check resolution input 
    if (not isinstance(resolution, (float, int, tuple))):
        raise TypeError("Provide resolution as single integer/float or as tuple.")
    if (isinstance(resolution, (float, int))):
        resolution = (resolution,resolution)
    else:
        if len(resolution) != 2:
            raise ValueError("Please provide resolution in (x,y) tuple format.")
    # Retrieve distance in x and y direction 
    dx = bbox[2] - bbox[0]
    dy = bbox[3] - bbox[1]
    # Check dx and dy are larger than resolution 
    if (dx < resolution[0]):
        raise ValueError("The specified resolution is larger than the bbox in x direction")
    if (dy < resolution[1]):
        raise ValueError("The specified resolution is larger than the bbox in y direction")    
    #-------------------------------------------------------------------------.
    # Retrieve corrections
    x_corr = (np.ceil(dx/resolution[0])*resolution[0] - dx)/2
    y_corr = (np.ceil(dy/resolution[1])*resolution[1] - dy)/2
    xmin = bbox[0]-x_corr
    ymin = bbox[1]-y_corr
    xmax = bbox[2]+x_corr
    ymax = bbox[3]+y_corr
    #-------------------------------------------------------------------------.
    # Check y is outside [-90, 90] in lat ...
    if (lonlat is True):
        if (ymin < -90): 
            thr = -90-ymin
            ymin = -90 
            ymax = ymax + thr
        if (ymax > 90): 
            thr = ymax -90 
            ymax = 90 
            ymin = ymin - thr
            if (ymin > -90):
                print("Impossible to adapt the bbox by extending it.")
                ymin = -90
    #-------------------------------------------------------------------------. 
    # Ensure that xmin and xmax stay inside [-180, 180] boundary
    if (xmin > 180): xmin = xmin - 360
    if (xmax > 180): xmax = xmax - 360
    if (xmin < 180): xmin = xmin + 360
    if (xmax < 180): xmax = xmax + 360
    #-------------------------------------------------------------------------. 
    return [xmin,ymin, xmax, ymax]
