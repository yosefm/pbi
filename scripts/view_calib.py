# -*- coding: utf-8 -*-
"""
The 3D-view calibration tool. Fingers crossed.

Created on Sun May 22 09:57:29 2016

@author: yosef
"""

import matplotlib.pyplot as pl, numpy as np
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    from mixintel.openptv import read_scene_config, intersect_known_points
    from optv.transforms import convert_arr_pixel_to_metric
    from tracer.spatial_geometry import rotx, roty, rotz, translate
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', 
        help="A YAML file with calibration and image properties.")
    args = parser.parse_args()
    
    yaml_args, cam_args, cpar = read_scene_config(args.config)
    print cpar.get_image_size()
    
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Known points:
    known = intersect_known_points([cam['known'] for cam in cam_args])
    ax.scatter(known[:,0], known[:,1], known[:,2])
    
    # Camera positions:
    pos = []
    glass_size = 100
    glass_plane = np.array([
            [glass_size, glass_size],
            [glass_size, -glass_size],
            [-glass_size, -glass_size],
            [-glass_size, glass_size],
            [glass_size, glass_size],
        ])

    for cam_spec in cam_args:
        pos.append(cam_spec['calib'].get_pos())
        prim = cam_spec['calib'].get_primary_point()
        angs = cam_spec['calib'].get_angles()
        
        # Find 3D positions of the image pixels and show as surface.
        h, w = cam_spec['image'].shape
        hs = np.linspace(0, h, 10)
        ws = np.linspace(0, w, 10)
        
        xs, ys = [np.float_(a) for a in np.meshgrid(ws, hs)]
        pixel_pos = np.hstack(( xs.reshape(-1,1), ys.reshape(-1,1) ))
        metric_pos = convert_arr_pixel_to_metric(pixel_pos, cpar)
        
        img_3d = np.hstack(
            (metric_pos , np.full((len(metric_pos), 1), -prim[2]*1), 
            np.ones((len(metric_pos), 1)) ) )
        #print img_3d
        
        # Rotate and translate to camera position/orientation:
        #trans = np.dot(rotz(angs[2]), np.dot(roty(angs[1]), rotx(angs[0])))
        trans = np.dot(rotx(angs[0]), np.dot(roty(angs[1]), rotz(angs[2])))
        trans = np.dot(translate(*pos[-1]), trans)
        pos_3d = np.dot(trans, img_3d[...,None])
        pos_3d = pos_3d[:3,:,-1].reshape(3, len(ys), len(xs))
        
        ax.plot_surface(*pos_3d[:3], color='r')#, facecolors=cam_spec['image'])
        
        # Glass delineation:
        glass_verts = np.hstack(( glass_plane,
            np.full((5,1), cam_spec['calib'].get_glass_vec()[-1]) ))
        ax.plot(*glass_verts.T)
        
    
    pos = np.array(pos)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2])
    ax.autoscale_view()
    ax.axis('scaled')
    #ax.set_aspect('equal') [-70, 70], [-70, 70], [0, 300]
    
    pl.show()
