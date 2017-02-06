# -*- coding: utf-8 -*-
"""
Sequence a PTV scene, creating the targets and rt_is files for each frame.

Created on Wed Jun  1 10:35:25 2016

@author: yosef
"""
from parallel_runner import PoolWorker

class FrameProc(PoolWorker):
    """
    A process class for doing all processing steps on a single frame: detection,
    correspondence, 3D determination, and writing it all out.
    """
    def __init__(self, tasks, command_pipe, results_queue, # pool comm
            cals, cam_args, seq_args, tpar, cpar, vpar):
        """
        Arguments:
        tasks, command_pipe, results_queue - see :module:``parallel_runner``.
        cals - a list of calibration objects.
        cam_args, seq_args - as obtained from the YAML config.
        tpar - a TargetParams object for detection.
        cpar - a ControlParams object with general scene parameters.
        vpar - a VolumeParameters object with correspondence parameters.
        """
        PoolWorker.__init__(self, tasks, command_pipe, results_queue)
        
        self._vpar = vpar
        self._cpar = cpar
        self._tpar = tpar
        self._cals = cals
        self._cams = cam_args
        self._seq = seq_args
    
    def job(self, frame):
        """
        Perform the processing.
        """
        print "processing frame %d" % frame
        
        detections = []
        for cix, cam_spec in enumerate(self._cams):
            img = pl.imread(self._seq['template'].format(
                cam=cix + 1, frame=frame))
            if args.method == 'large':
                targs = detect_large_particles(
                    img, approx_size=self._seq['radius'],
                    peak_thresh=cam_spec['peak_threshold'])
            elif args.method == 'dog':
                targs = detect_blobs(img, thresh=self._seq['threshold'])
            else:
                hp = simple_highpass(img, self._cpar)
                targs = target_recognition(
                    hp, self._tpar, cix, self._cpar)
            detections.append(targs)
        
        if any([len(det) == 0 for det in detections]):
            return False
        
        # Corresp. + positions.
        sets, corresp, _ = correspondences(
            detections, self._cals, self._vpar, self._cpar)
        
        # Save targets only after they've been modified:
        for cix in xrange(len(self._cams)):
            detections[cix].write(
                seq['targets_template'].format(cam=cix + 1), frame)
            
        # Distinction between quad/trip irrelevant here.
        sets = np.concatenate(sets, axis=1)
        corresp = np.concatenate(corresp, axis=1).T
        
        flat = []
        for cam, cal in enumerate(self._cals):
            unused = (sets[cam] == -999)
            metric = convert_arr_pixel_to_metric(sets[cam], self._cpar)
            flat.append(distorted_to_flat(metric, cal))
            flat[-1][unused] = -999
        
        flat = np.array(flat)
        pos, rcm = point_positions(
            flat.transpose(1,0,2), self._cpar, self._cals)
        
        # Save rt_is
        rt_is = open(seq['output_template'].format(frame=frame), 'w')
        rt_is.write(str(pos.shape[0]) + '\n')
        for pix, pt in enumerate(pos):
            pt_args = (pix + 1,) + tuple(pt) + tuple(corresp[pix])
            rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)
        rt_is.close()
        
        return True
            
if __name__ == "__main__":
    import argparse, time
    import matplotlib.pyplot as pl, numpy as np
    from multiprocessing import Pipe, Queue
    from Queue import Empty
    
    from mixintel.openptv import read_scene_config, simple_highpass
    from mixintel.detection import detect_large_particles, detect_blobs
    from optv.parameters import VolumeParams
    from optv.segmentation import target_recognition
    from optv.transforms import convert_arr_pixel_to_metric, \
        distorted_to_flat
    
    from calib import correspondences, point_positions
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', 
        help="A YAML file with calibration and image properties.")
    parser.add_argument(
        '--method', '-m', default='default', 
        choices=['default', 'large', 'dog'],
        help="Change detection method to one suitable for large particles.")
    parser.add_argument(
        '--procs', '-p', default=4, type=int,
        help="Number of parallel processes.")
    args = parser.parse_args()
    
    yaml_args, cam_args, cpar = read_scene_config(args.config)
    vpar = VolumeParams(**yaml_args['correspondences'])
    seq = yaml_args['sequence']
    cals = [cam_spec['calib'] for cam_spec in cam_args]
    
    # Parallel processing setup.
    # Maximum queue size mandates blocking until a slot is free
    results = Queue()
    num_procs = args.procs
    tasks = Queue(num_procs*2)
    w = []
    
    for p in xrange(num_procs):
        pside, cside = Pipe()
        t = FrameProc(
            tasks, cside, results,cals, cam_args, seq, yaml_args['targ_par'], 
            cpar, vpar)
        w.append((t, pside))
        t.start()

        time.sleep(0.5)
    
    got_res = 0
    for frame in xrange(seq['first'], seq['last'] + 1):
        while tasks.qsize() > num_procs * 1.5:
            time.sleep(0.005)
        tasks.put(frame)
        try:
            res = results.get_nowait()
            got_res += 1
        except Empty:
            pass
    
    while got_res < seq['last'] - seq['first']:
        try:
            res = results.get_nowait()
            got_res += 1
        except Empty:
            pass
        
    for p in w:
        p[0].terminate()
