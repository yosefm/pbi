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
    5.04204635e+00   3.86231016e+00   2.77770502e+02   2.61769309e-01
  -9.12249692e-02   1.92978937e-03   1.18159200e+00  -1.21095361e+00
   6.21660950e+01  -4.00000000e-05  -1.87130515e-06   5.05303698e-07
   3.35195338e-06   2.97641313e-06  -1.43296649e+01   7.13349185e+00
   2.90568152e+02   2.66614557e-01   4.36536443e-01  -5.76774577e-02
   2.29507910e-01  -4.63904120e-01   6.73813006e+01   7.69919802e-06
   7.39193614e-06   5.76784795e-07   2.36766763e-07   6.07396975e-07
   5.40582643e+00  -1.40000000e+01   2.90906885e+02  -4.28121109e-01
  -2.44308756e-01  -1.09248304e-01   8.36752253e-02   1.60230965e+00
   6.49751825e+01   1.95185477e-04  -6.07585960e-06   7.33705397e-07
  -6.20982499e-04  -2.00000000e-04  -6.89282392e+00  -1.91334120e+00
   2.94293882e+02  -4.19251658e-01   3.92636910e-01   1.06559767e-01
   2.63631708e+00   4.37072456e+00   6.61127565e+01  -1.67522214e-05
  -6.84526104e-06   2.67730232e-07  -6.38324691e-04   7.99122959e-05
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
            glass_vec = np.r_[0., 0.,  118.]
                
            cal = gen_calib(inters, R, angs, glass_vec, prim, rad, decent)
            
            name = args.output % (cam + 1)
            cal.write(name + '.ori', name + '.addpar')
    