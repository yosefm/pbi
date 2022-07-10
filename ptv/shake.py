# -*- coding: utf-8 -*-
"""
BOOM shake shake shake the room!!!

Fine-tune calibration using the "shaking" method of comparing 3D positions 
obtained with existing calibration to their 2D projections. It's a kind of a 
feedback step over the normal calibration with known points.

Created on Sun Jan 31 13:42:18 2016

@author: Yosef Meller
"""

if __name__ == "__main__":
    import argparse, yaml, numpy as np
    
    from optv.orientation import full_calibration
    from optv.parameters import ControlParams
    
    from optv.tracking_framebuf import TargetArray, Frame
    from optv.calibration import Calibration
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to configuration YAML file.")
    parser.add_argument('--dry-run', '-d', action='store_true', default=False,
        help="Don't write ori/addpar files with results.")
    parser.add_argument('--output', '-o', 
        help="Template calibration filename with one %%d for camera number." + \
        "default is to overwrite initial guess. Filename extension is added.")
    args = parser.parse_args()
    
    yaml_args = yaml.load(open(args.config,'r'), yaml.CLoader)
    yaml_args.setdefault('skip', 1)
    
    # Generate initial-guess calibration objects. These get overwritten by
    # the optimizer's target function.
    cal_args = yaml_args['initial']
    calibs = []
    active = []
    
    for cam_data in cal_args:
        cl = Calibration()
        cl.from_file(cam_data['ori_file'].encode(), cam_data['addpar_file'].encode())
        
        if cam_data['free']:
            active.append(len(calibs))
        calibs.append(cl)
    
    scene_args = yaml_args['scene']
    num_cams = len(cal_args)
    # print(f'Number of cameras is {num_cams}')
    scene_args['cams'] = num_cams
    cpar = ControlParams(**scene_args)
    # print(f' and from cpar is {cpar.get_num_cams()}')
    
    targ_files = [yaml_args['target_template'].encode() % c for c in \
        range(num_cams)]

    print(targ_files)

    # Iterate over frames, loading the big lists of 3D positions and 
    # respective detections.
    all_known = []
    all_detected = [[] for c in range(cpar.get_num_cams())]
    for frm_num in range(yaml_args['first'], yaml_args['last'] + 1, 
                          yaml_args['skip']):
        frame = Frame(cpar.get_num_cams(), 
            corres_file_base=yaml_args['corres_file_base'].encode(), 
            linkage_file_base=yaml_args['linkage_file_base'].encode(),
            target_file_base=targ_files, frame_num=frm_num)
            
        all_known.append(frame.positions())
        for cam in active:
            all_detected[cam].append(frame.target_positions_for_camera(cam))
    
    # Make into the format needed for full_calibration.
    all_known = np.vstack(all_known)
    
    # Calibrate each camera accordingly.
    for cam in active:
        detects = np.vstack(all_detected[cam])
        assert detects.shape[0] == all_known.shape[0]
        
        have_targets = ~np.isnan(detects[:,0])
        used_detects = detects[have_targets,:]
        used_known = all_known[have_targets,:]
        
        targs = TargetArray(len(used_detects))
        
        for tix in range(len(used_detects)):
            targ = targs[tix]
            targ.set_pnr(tix)
            targ.set_pos(used_detects[tix])
        
        residuals = full_calibration(calibs[cam], used_known, targs, cpar)
        
        if args.dry_run:
            print(("Camera %d" % (cam + 1)))
            print((calibs[cam].get_pos()))
            print((calibs[cam].get_angles()))
        else:
            if args.output is None:
                ori = cal_args[cam]['ori_file']
                distort = cal_args[cam]['addpar_file']
            else:
                ori = args.output % (cam + 1) + '.ori'
                distort = args.output % (cam + 1) + '.addpar'
            
            calibs[cam].write(ori.encode(), distort.encode())
            
# I don't like the dangling end of Python scripts... Do I have OCD?
