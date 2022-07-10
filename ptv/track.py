#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Perform a tracking pass on the output of the sequencing pass for a scene.

Created on Mon Apr 24 14:21:46 2017

@author: yosef
"""

if __name__ == "__main__":
    import yaml
    from optv.calibration import Calibration
    from optv.parameters import ControlParams, VolumeParams, TrackingParams,\
        SequenceParams
    from optv.tracker import Tracker
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', 
        help="A YAML file with tracking and related parameters.")
    args = parser.parse_args()
    
    with open(args.config) as f:
        yaml_conf = yaml.load(f, yaml.CLoader)
    seq_cfg = yaml_conf['sequence']
    
    cals = []
    img_base = []
    for cix, cam_spec in enumerate(yaml_conf['cameras']):
        cam_spec.setdefault('addpar_file', None)
        cal = Calibration()
        cal.from_file(cam_spec['ori_file'], cam_spec['addpar_file'])
        cals.append(cal)
        img_base.append(seq_cfg['targets_template'].format(cam=cix + 1))
        
    cpar = ControlParams(len(yaml_conf['cameras']), **yaml_conf['scene'])
    vpar = VolumeParams(**yaml_conf['correspondences'])
    tpar = TrackingParams(**yaml_conf['tracking'])
    spar = SequenceParams(
        image_base=img_base,
        frame_range=(seq_cfg['first'], seq_cfg['last']))
    
    framebuf_naming = {
        'corres': 'res/particles',
        'linkage': 'res/linkage',
        'prio': 'res/whatever'
    }
    for ftype in ['corres', 'linkage', 'prio']:
        if ftype + '_prefix' in yaml_conf['tracking']:
            framebuf_naming[ftype] = yaml_conf['tracking'][ftype + '_prefix']
    
    tracker = Tracker(cpar, vpar, tpar, spar, cals, framebuf_naming)
    tracker.full_forward()
    
    # Since tracking misbehaves sometimes on a dirty targets database,
    # we allow saving a copy of the tracked targets, so that one can nuke the 
    # target files in the original and start over, still having the results of 
    # this tracking run saved somewhere safe.
    if 'targets_copy' in seq_cfg:
        import shutil, os
        if not os.path.exists(seq_cfg['targets_copy']):
            os.mkdir(seq_cfg['targets_copy'])
        
        for base in img_base:
            for frm in range(seq_cfg['first'], seq_cfg['last']):
                src = base + str(frm) + '_targets'
                shutil.copy(src, seq_cfg['targets_copy'])
    
