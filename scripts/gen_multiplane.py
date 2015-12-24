# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:34:30 2015

@author: yosef
"""

single_yaml = """
target:
    image: {path}/{plane}{cam}.tif
    number: {camn}
    ori_file: {path}/{plane}{cam}.tif.ori
    addpar_file: {path}/{plane}{cam}.tif.addpar
    known_points: {path}/points_{plane}.txt
    manual_detection_points: [ 1, 5, 16, 20]
    manual_detection_file: {path}/{plane}{cam}_man.txt
    detection_par_file: parameters/targ_rec.par

scene:
    flags: hp, headers
    image_size: [ 1280, 1024 ]
    pixel_size: [ 0.014, 0.014 ]

    # Multimedia parameters:
    cam_side_n: 1  # air
    object_side_n: 1.33  # perspex
    wall_ns: [ 1.43 ]
    wall_thicks: [ 5 ]
"""

multi_yaml = """
- image: {path}/{plane}{cam}.tif
  ori_file: cal_single/multi{cam}.tif.ori
  addpar_file: cal_single/multi{cam}.tif.addpar

"""

if __name__ == "__main__":
    import os.path, shutil, numpy as np
    
    import argparse
    parser = argparse.ArgumentParser(
        usage="Assumes symmetric planes around central.")
    parser.add_argument('--output', '-o', default="cal_mp",
        help="directory path for generated files")
    parser.add_argument('--num-cams', '-n', type=int, default=4)
    parser.add_argument('--points-file', '-p', default='points.txt',
        help="Table of known points position (calblock)")
    parser.add_argument('--shift', '-s', type=float, default=9.0)
    parser.add_argument('--clobber', '-c', action='store_true', default=False,
        help="Overwrite existing files")
    parser.add_argument('--central', '-t', default='cam',
        help="Name of central plane (default 'cam') used as template.")
    parser.add_argument('planes', nargs='*',
        help="Names of added planes, in order from -Z to +Z.")
    args = parser.parse_args()
    
    known_points = np.loadtxt(os.path.join(args.output, args.points_file))
    
    # Shifts discard the central plane which is assumed to exist.
    each_side = len(args.planes)/2 # integer division
    plane_shifts = np.arange(-each_side, each_side + 1) * args.shift
    plane_shifts = np.r_[plane_shifts[:each_side], plane_shifts[-each_side:]]
    
    for plane, shift in zip(args.planes, plane_shifts):
        # Single-plane, single camera calibration tool configuration,
        # needed for generating the fix/crd point files.
        for camn in xrange(args.num_cams):
            cam = camn + 1
            fname = os.path.join(args.output, "%s%d.yaml" % (plane, cam))
        
            if os.path.exists(fname) and not args.clobber:
                continue
        
            cfg = single_yaml.format(path=args.output, camn=camn, cam=cam, 
                plane=plane)
            cfg_file = open(fname, 'w')
            cfg_file.write(cfg)
            cfg_file.close()
    
            # Template files for the singe camera/plane calibrations: 
            # ori, addpar,
            ori_tmpl = os.path.join(args.output, "%s%d.tif.ori")
            addpar_tmpl = os.path.join(args.output, "%s%d.tif.addpar")
            
            shutil.copy(ori_tmpl % (args.central, cam), 
                ori_tmpl % (plane, cam))
            shutil.copy(addpar_tmpl % (args.central, cam), 
                addpar_tmpl % (plane, cam))
    
        # Per-plane points files
        fname = os.path.join(args.output, "points_wf_%s.txt" % plane)
        if not os.path.exists(fname) or args.clobber:
            plane_pts = known_points.copy()
            plane_pts[:,3] += shift
            np.savetxt(fname, plane_pts, fmt="%10.5f")
    
        # per-plane epi-checker config 
        fname = os.path.join(args.output, "multi/%s.yaml" % plane)
        if os.path.exists(fname) and not args.clobber:
                continue
        
        epi_yaml = []
        for camn in xrange(args.num_cams):
            epi_yaml.append(
                multi_yaml.format(path=args.output, plane=plane, cam=camn + 1))
        
        epi_file = open(fname, 'w')
        epi_file.writelines(epi_yaml)
        epi_file.close()
