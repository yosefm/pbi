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
    -1.34906106e+00  -1.68920717e+00   2.59914505e+02   1.68256112e-01
   2.93567359e+00   6.17444626e-02   8.25448983e-03   1.80558328e-02
   7.39321639e+01   1.17815075e-05  -2.24319799e-06   8.53523369e-07
   5.37055835e-06   5.16253575e-06   1.96945223e+00  -2.37223034e+00
   2.51087355e+02   1.68934309e-01   3.30000320e+00   7.73814578e-04
   2.29450271e-02  -2.36679270e-02   7.16281721e+01  -5.59570714e-06
  -5.34243556e-06   1.03781533e-06   4.88551757e-06  -3.72863589e-06
   1.37429948e+01  -7.23542058e+00   2.95607146e+02  -1.94110432e-01
  -3.07648510e-01  -5.57567585e-02   4.26295284e-02  -2.14068283e-02
   8.80756365e+01   1.73927074e-07  -7.90015984e-06   3.46946016e-07
   4.25167293e-06  -7.37563249e-06  -5.22094776e+00  -3.01885542e+00
   2.82485175e+02  -1.12318483e-01   1.58363078e-01   1.64894075e-02
  -9.58576068e-03  -1.54172047e-02   7.85690041e+01  -6.68780743e-06
  -4.23788233e-06   1.03967633e-06  -2.11857070e-06  -4.44989977e-06
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
    