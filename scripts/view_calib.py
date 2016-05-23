# -*- coding: utf-8 -*-
"""
The 3D-view calibration tool. Fingers crossed.

Created on Sun May 22 09:57:29 2016

@author: yosef
"""

import matplotlib.pyplot as pl, numpy as np
from mpl_toolkits.mplot3d import Axes3D

from tracer.assembly import Assembly
from tracer.object import AssembledObject
from tracer.surface import Surface
from tracer.flat_surface import RectPlateGM
from tracer import optics_callables as opt
from tracer.tracer_engine import TracerEngine

class GlassPlate(AssembledObject):
    def __init__(self, width, height, ns, nb, nd, thickness, transform=None):
        """
        Implement a simple glass optical object with two refraction surfaces 
        whose refraction indices on all sides are given by the user.
        
        Arguments:
        width, height - local x, y dimentions of plate, respectively.
        ns, nb, nd - refraction indices at source, bulk and destination of rays
            respectively. 'source' is the side where rays would come from, so 
            consider transforms when putting in the numbers.
        thickness - distance between interfaces.
        transform - a 4x4 homogeneous transformation matrix for placing the
            object in space. Initial placement is that the destination side is 
            on the local XY plane and the source side is shifted toward +Z.
        """
        backside = Surface(RectPlateGM(width, height),
            opt.RefractiveHomogenous(nb, nd))
        frontside = Surface(RectPlateGM(width, height),
            opt.RefractiveHomogenous(ns, nb), location=np.r_[0., 0., thickness]
        )
        AssembledObject.__init__(self, surfs=[backside, frontside], 
            transform=transform)
       
def show_rays(axes, tree, escaping_len):
    """
    Given the tree data structure from the ray tracing engine, 
    3D-plot the rays.
    
    Arguments:
    axes - a matplotlib Axes3D object on which to draw the lines.
    tree - a tree of rays, as constructed by the tracer engine
    escaping_len - the length of the arrow indicating the direction of rays
        that don't intersect any surface (leaf rays).
    """
    for level in xrange(tree.num_bunds()):
        start_rays = tree[level]
        se = start_rays.get_energy()
        non_degenerate = (se != 0) & (start_rays.get_directions()[2,:] < 0)
        sv = start_rays.get_vertices()[:,non_degenerate]
        sd = start_rays.get_directions()[:,non_degenerate]
        
        if level == tree.num_bunds() - 1:
            # Make endpoints for escaping rays
            ev = sv + sd*escaping_len
        else:
            end_rays = tree[level + 1]
            escaping = ~np.any(
                np.arange(sv.shape[1]) == end_rays.get_parents()[:,None], 0)
            escaping_endpoints = sv[:,escaping] + sd[:,escaping]*escaping_len
            
            ev = np.hstack((end_rays.get_vertices(), escaping_endpoints))
        
        for seg in zip(sv.T, ev.T):
            axes.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                      [seg[0][2], seg[1][2]], c='y')
        
if __name__ == "__main__":
    from mixintel.openptv import read_scene_config, intersect_known_points
    from optv.transforms import convert_arr_pixel_to_metric, correct_arr_brown_affine
    
    from tracer.spatial_geometry import rotx, roty, rotz, translate
    from tracer.ray_bundle import RayBundle
    
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
    ax.scatter(known[:,0], known[:,1], known[:,2], s=0.5)
    
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
        
    ray_sources = RayBundle.empty_bund()
    mm = cpar.get_multimedia_params()
    
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
            (metric_pos - prim[:2], np.full((len(metric_pos), 1), -prim[2]), 
            np.ones((len(metric_pos), 1)) ) )
        #print img_3d
        
        # Rotate and translate to camera position/orientation:
        #trans = np.dot(rotz(angs[2]), np.dot(roty(angs[1]), rotx(angs[0])))
        rot = np.dot(rotx(angs[0]), np.dot(roty(angs[1]), rotz(angs[2])))
        trans = np.dot(translate(*pos[-1]), rot)
        pos_3d = np.dot(trans, img_3d[...,None])
        pos_3d = pos_3d[:3,:,-1].reshape(3, len(ys), len(xs))
        
        ax.plot_surface(*pos_3d[:3], color='r')#, facecolors=cam_spec['image'])
        
        # Glass delineation:
        glass_verts = np.hstack(( glass_plane,
            np.full((5,1), cam_spec['calib'].get_glass_vec()[-1]) ))
        ax.plot(*glass_verts.T)
        
        # Generate ray bundle:
        targs = np.array([t.pos() for t in cam_spec['targs']])
        metric = convert_arr_pixel_to_metric(targs, cpar) - prim[:2]
        flat = correct_arr_brown_affine(metric, cam_spec['calib'])
        
        detect_local = np.hstack((flat, np.full((len(targs), 1), -prim[2]),
            np.ones((len(targs), 1)) ))
        detect = np.dot(rot, detect_local[...,None])
        ray_dirs = detect[:3,:,-1]
        ray_dirs /= np.linalg.norm(ray_dirs, axis=0)
        ray_sources += RayBundle(vertices=np.tile(pos[-1], (ray_dirs.shape[1], 1)).T,
            directions=ray_dirs, energy=np.ones(ray_dirs.shape[1]),
            ref_index=np.repeat(mm.get_n1(), ray_dirs.shape[1]))
    
    pos = np.array(pos)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2])
    
    # Prepare ray tracing scene:
    glass = GlassPlate( glass_size*2, glass_size*2, 
        mm.get_n1(), mm.get_n2()[0], mm.get_n3(), mm.get_d()[0], 
        transform=translate(*cam_spec['calib'].get_glass_vec()) )
    asm = Assembly(objects=[glass])
    
    # Trace the scene and show rays:
    tr = TracerEngine(asm)
    tr.ray_tracer(ray_sources, 5, 0.1)
    show_rays(ax, tr.tree, 200.)
    
    ax.autoscale_view()
    ax.axis('scaled')
    #ax.set_aspect('equal') [-70, 70], [-70, 70], [0, 300]
    
    pl.show()
