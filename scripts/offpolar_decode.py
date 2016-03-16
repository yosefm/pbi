# -*- coding: utf-8 -*-
"""
Convert offset-polar representation of calibration, as output z.B. by 
evo_corres.py, to the normal cartesian coordinates.

Created on Sun Mar 13 11:17:58 2016

@author: yosef
"""

if __name__ == "__main__":
    import numpy as np
    from mixintel.evolution import get_pos, gen_calib
    from optv.calibration import Calibration
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', 
        help="Outpur format string, one %%d for camera number w/o extension.")
    args = parser.parse_args()
    
    num_cams = 4
    offpol_str = """
    -8.71452648e+00   3.32105768e+00   2.75635699e+02   1.65177068e-01
   2.90896993e+00   2.73141190e-02  -3.12272672e-02  -4.50332810e-03
   7.15955088e+01  -1.19808850e-05  -1.08873933e-05   8.19935416e-07
  -4.51914405e-06  -7.94899254e-06   3.35275398e+00   2.96358790e+00
   2.67919599e+02   1.67587712e-01   3.30283566e+00   1.93496616e-02
   3.11839277e-02  -4.51951264e-02   7.00215546e+01   1.98286189e-05
  -6.45510160e-06   5.18962009e-07   6.80435850e-06  -1.86538988e-06
   3.65900733e+00   4.37711390e+00   2.66513480e+02  -1.72861692e-01
  -2.84251727e-01  -5.98036614e-02  -3.97090183e-02  -2.09196974e-03
   8.11191122e+01   2.71625620e-06   3.89075540e-06  -5.29433489e-08
  -2.22867633e-06  -1.00000000e-05  -6.41467949e+00   7.27052530e+00
   2.81652898e+02  -7.05229919e-02   1.83514408e-01   2.10095500e-02
   3.74549615e-02  -1.06702506e-02   7.93568618e+01   6.94696440e-06
  -6.44516664e-06   6.95389309e-07   3.27353982e-07   7.90607949e-07
    """
    offpol = np.array([float(p) for p in offpol_str.split()
        ]).reshape(num_cams, -1)
    
    for cam in xrange(num_cams):
        angs = offpol[cam,3:6]
        inters = np.zeros(3)
        inters[:2] = offpol[cam,:2]
        R = offpol[cam,2]
        pos = get_pos(inters, R, angs)
        prim = offpol[cam,6:9]
        rad = offpol[cam,9:12]
        decent = offpol[cam,12:14]
        
        print
        print "camera %d" % (cam + 1)
        print "----------"
        print "pos/ang:"
        print "%.8f %.8f %.8f" % tuple(pos)
        print "%.8f %.8f %.8f" % tuple(angs)
        print
        print "internal: %.8f %.8f %.8f" % tuple(prim)
        print "radial distortion: %.8f %.8f %.8f" % tuple(rad)
        print "decentering: %.8f %.8f" % tuple(decent)
        
        if args.output is not None:
            if cam < 2:
                glass_vec = np.r_[0., 0., -100.]
            else:
                glass_vec = np.r_[0., 0.,  100.]
                
            cal = gen_calib(inters, R, angs, glass_vec, prim, rad, decent)
            
            name = args.output % (cam + 1)
            cal.write(name + '.ori', name + '.addpar')
    