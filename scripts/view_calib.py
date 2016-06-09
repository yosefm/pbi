# -*- coding: utf-8 -*-
"""
The 3D-view calibration tool. Fingers crossed.

Created on Sun May 22 09:57:29 2016

@author: yosef
"""

import numpy as np
import traitsui.api as tui

from tracer.assembly import Assembly
from tracer.object import AssembledObject
from tracer.surface import Surface
from tracer.flat_surface import RectPlateGM
from tracer import optics_callables as opt
#from tracer.tracer_engine import TracerEngine
from tracer.mayavi_ui.scene_view import TracerScene

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

class PTVScene(TracerScene):
    view = tui.View(TracerScene.scene_view_item(700, 700))
    
    def __init__(self, cam_args, cpar, glass_size=100):
        """
        Sets up initial trace and constant scenery.
        
        Arguments:
        cam_args - list, camera argument dicts as loaded from YAML config.
        cpar - a ControlParams object with general scene information.
        glass_size - width and height of intervening glass to show.
        """                        
        # Auxiliary surfaces and ray sources:
        pos = []
        ray_sources = RayBundle.empty_bund()
        colors = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0.5, 0., 0.5)]
        mm = cpar.get_multimedia_params()
        
        for cam_spec, color in zip(cam_args, colors):
            pos.append(cam_spec['calib'].get_pos())
            prim = cam_spec['calib'].get_primary_point()
            angs = cam_spec['calib'].get_angles()
            
            # Find 3D positions of the image pixels and show as surface.
            h, w = cam_spec['image_data'].shape
            hs = np.linspace(0, h, 5)
            ws = np.linspace(0, w, 5)
            
            xs, ys = [np.float_(a) for a in np.meshgrid(ws, hs)]
            pixel_pos = np.hstack(( xs.reshape(-1,1), ys.reshape(-1,1) ))
            metric_pos = convert_arr_pixel_to_metric(pixel_pos, cpar)
            
            img_3d = np.hstack(
                (metric_pos - prim[:2], np.full((len(metric_pos), 1), -prim[2]), 
                np.ones((len(metric_pos), 1)) ) )
            
            # Rotate and translate to camera position/orientation:
            #trans = np.dot(rotz(angs[2]), np.dot(roty(angs[1]), rotx(angs[0])))
            rot = np.dot(rotx(angs[0]), np.dot(roty(angs[1]), rotz(angs[2])))
            trans = np.dot(translate(*pos[-1]), rot)
            pos_3d = np.dot(trans, img_3d[...,None])
            pos_3d = pos_3d[:3,:,-1].reshape(3, len(ys), len(xs))
            
            self._scene.mlab.mesh(*pos_3d, color=color)
                    
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
        
        # Prepare ray tracing scene:
        glass = GlassPlate( glass_size*2, glass_size*2, 
            mm.get_n1(), mm.get_n2()[0], mm.get_n3(), mm.get_d()[0], 
            transform=translate(*cam_spec['calib'].get_glass_vec()) )
        assembly = Assembly(objects=[glass])
        
        sel = lambda sr: sr.get_directions()[2,:] < 0
        for surf in assembly.get_surfaces():
            surf.resolution = 0.1
            surf.opacity = 0.2

        TracerScene.__init__(self, assembly, ray_sources, escaping=200., 
            ray_selector=sel)
        self.set_background((0., 0.5, 1.))
        self._scene.mlab.points3d(pos[:,0], pos[:,1], pos[:,2], 
            color=(1., 0., 0.), scale_factor=1.)
        
        known_points = intersect_known_points([cam['known'] for cam in cam_args])
        self._scene.mlab.points3d(known_points[:,0], known_points[:,1], 
            known_points[:,2], color=(1., 0., 0.), scale_factor=0.5)

        
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
        
    scn = PTVScene(cam_args, cpar)
    scn.set_trace_control(max_iter=5, min_energy=0.1)
    scn.configure_traits()
    
