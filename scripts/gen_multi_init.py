# -*- coding: utf-8 -*-
"""
Generates an initial guess for multiplane calibration that is hopefully
different than each of the single-plane calibrations. This should
prevent the case of the optimization staying in a local minimum that 
satysfies one plane perfectly and neglects the others.

Created on Wed Dec  2 13:38:25 2015

@author: yosef
"""
from optv.calibration import Calibration

if __name__ == "__main__":
    import argparse, numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('cam', type=int, help="Camera number")
    parser.add_argument('planes', nargs='*',
        help="Names of participating planes")
    args = parser.parse_args()
    
    path="cal_single/"
    cal_plane = Calibration()
    cal_final = Calibration()
    cal_final.set_affine_trans(np.r_[1, 0])
    
    # Accumulators:
    pos = np.zeros(3)
    angles = np.zeros(3)
    primp = np.zeros(3)
    radial = np.zeros(3)
    decent = np.zeros(2)
    
    for plane in args.planes:
        ori = "{path}/{plane}{cam}.tif.ori".format(path=path, plane=plane,
            cam = args.cam)
        addpar = "{path}/{plane}{cam}.tif.addpar".format(path=path, plane=plane,
            cam = args.cam)
        cal_plane.from_file(ori, addpar)
        
        pos += cal_plane.get_pos()
        angles += cal_plane.get_angles()
        primp += cal_plane.get_primary_point()
        radial += cal_plane.get_radial_distortion()
        decent += cal_plane.get_decentering()
    
    num_planes = len(args.planes)
    cal_final.from_file(ori, addpar) # Just any plane, to get the glass vector.
    
    cal_final.set_pos(pos/num_planes)
    cal_final.set_angles(angles/num_planes)
    cal_final.set_primary_point(primp/num_planes)
    cal_final.set_radial_distortion(radial/num_planes)
    cal_final.set_decentering(decent/num_planes)
    
    ori = "{path}/multi{cam}.tif.ori".format(path=path, cam=args.cam)
    addpar = "{path}/multi{cam}.tif.addpar".format(path=path, cam=args.cam)
    cal_final.write(ori, addpar)
