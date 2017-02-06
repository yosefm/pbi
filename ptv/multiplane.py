# -*- coding: utf-8 -*-
"""
Multiplane calibration, unsupervised. Collects the detections and matched fixed
points (3D known positions) from each plane, then performs the final 
orientation (orient_v3) with the combined sets.

Created on Sun Nov 29 14:34:12 2015

@author: yosef
"""

if __name__ == "__main__":
    import sys, argparse, yaml, numpy as np
    from calib import full_calibration
    
    from optv.tracking_framebuf import TargetArray
    from optv.calibration import Calibration
    from optv.parameters import ControlParams
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to configuration YAML file.")
    parser.add_argument('--dry-run', '-d', action='store_true', default=False,
        help="Don't overwrite ori/addpar files with results.")
    parser.add_argument('--renum', '-r', action='store_true', default=False,
        help="Rewrite the fix/crd files renumbered and stop.")
    args = parser.parse_args()
    
    yaml_args = yaml.load(file(args.config))
    
    # Load fix/crd, renumbering along the way.
    all_known = []
    all_detected = []
    
    for plane in yaml_args['planes']:
        known = np.loadtxt(plane['known'])
        detected = np.loadtxt(plane['detected'])
        
        if np.any(detected == -999):
            raise ValueError(("Using undetected points in {} will cause " + 
                "silliness. Quitting.").format(plane['detected']))
        
        num_known = len(known)
        num_detect = len(detected)
        
        if num_known != num_detect:
            raise ValueError("Number of detected points (%d) does not match" +\
            " number of known points (%d) for %s, %s" % \
            (num_known, num_detect, plane['known'],  plane['detected']))
        
        if len(all_known) > 0:
            detected[:,0] = all_detected[-1][-1,0] + 1 + np.arange(len(detected))
        
            # Save renumbered file, for PyPTV compatibility:
            if args.renum:
                known[:,0] = all_known[-1][-1,0] + 1 + np.arange(len(known))
                np.savetxt(plane['known'], known, fmt="%10.5f")
                np.savetxt(plane['detected'], detected, fmt="%9.5f")

        all_known.append(known)
        all_detected.append(detected)
    
    if args.renum:
        sys.exit(0)
    
    # Make into the format needed for full_calibration.
    all_known = np.vstack(all_known)[:,1:]
    all_detected = np.vstack(all_detected)
    
    targs = TargetArray(len(all_detected))
    for tix in xrange(len(all_detected)):
        targ = targs[tix]
        det = all_detected[tix]
        
        targ.set_pnr(tix)
        targ.set_pos(det[1:])
    
    # Load ori files and whatever else is needed over the point sets.
    cal = Calibration()
    cal.from_file(yaml_args['target']['ori_file'], 
        yaml_args['target']['addpar_file'])
    
    control_args = yaml_args['scene']
    control = ControlParams(1)
    control.set_hp_flag( 1 if 'hp' in control_args['flags'] else 0)
    control.set_allCam_flag( 1 if 'allcam' in control_args['flags'] else 0)
    control.set_tiff_flag( 1 if 'headers' in control_args['flags'] else 0)
    control.set_imx(control_args['image_size'][0])
    control.set_imy(control_args['image_size'][1])
    control.set_pix_x(control_args['pixel_size'][0])
    control.set_pix_y(control_args['pixel_size'][1])
    control.set_chfield(0)
    
    layers = control.get_multimedia_params()
    layers.set_lut(0)
    layers.set_n1(control_args['cam_side_n'])
    layers.set_n2(control_args['wall_ns'])
    layers.set_n3(control_args['object_side_n'])
    layers.set_d(control_args['wall_thicks'])
    layers.set_nlay(len(control_args['wall_ns']))
    
    # Do the thing.
    residuals, used = full_calibration(cal, all_known, targs, control)
    
    # Save results.
    if not args.dry_run:
        cal.write(yaml_args['target']['ori_file'], 
            yaml_args['target']['addpar_file'])
