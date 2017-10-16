# -*- coding: utf-8 -*-
"""
Sequence a PTV scene, creating the targets and rt_is files for each frame.

Created on Wed Jun  1 10:35:25 2016

@author: yosef
"""
import numpy as np, matplotlib.pyplot as pl
from parallel_runner import PoolWorker

from optv.correspondences import correspondences, MatchedCoords
from optv.segmentation import target_recognition
from optv.orientation import point_positions

from util.openptv import simple_highpass
from util.detection import detect_large_particles, detect_blobs

class FrameProc(PoolWorker):
    """
    A process class for doing all processing steps on a single frame: detection,
    correspondence, 3D determination, and writing it all out.
    """
    def __init__(self, tasks, command_pipe, results_queue, # pool comm
            cals, cam_args, seq_args, tpar, cpar, vpar, report=False):
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
        self._report = report
    
    def job(self, frame):
        """
        Perform the processing.
        """
        print "processing frame %d" % frame
        
        detections = []
        corrected = []
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
            
            targs.sort_y()
            detections.append(targs)
            corrected.append(MatchedCoords(targs, self._cpar, self._cals[cix]))
        
        if any([len(det) == 0 for det in detections]):
            return False
        
        # Corresp. + positions.
        sets, corresp, _ = correspondences(
            detections, corrected, self._cals, self._vpar, self._cpar)
        
        # Save targets only after they've been modified:
        for cix in xrange(len(self._cams)):
            detections[cix].write(
                seq['targets_template'].format(cam=cix + 1), frame)
        
        if self._report:
            print "Frame " + str(frame) + " had " \
            + repr([s.shape[1] for s in sets]) + " correspondences."
        
        # Distinction between quad/trip irrelevant here.
        sets = np.concatenate(sets, axis=1)
        corresp = np.concatenate(corresp, axis=1)
        
        flat = np.array([corrected[cix].get_by_pnrs(corresp[cix]) \
            for cix in xrange(len(self._cals))])
        pos, rcm = point_positions(
            flat.transpose(1,0,2), self._cpar, self._cals)
        
        # Save rt_is
        rt_is = open(seq['output_template'].format(frame=frame), 'w')
        rt_is.write(str(pos.shape[0]) + '\n')
        for pix, pt in enumerate(pos):
            pt_args = (pix + 1,) + tuple(pt) + tuple(corresp[:,pix])
            rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)
        rt_is.close()
            
        return True
            
if __name__ == "__main__":
    import argparse, time
    from multiprocessing import Pipe, Queue
    from Queue import Empty
    
    from util.openptv import read_scene_config
    from optv.parameters import VolumeParams
    
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
    parser.add_argument(
            '--report', '-r', action='store_true', default=False,
            help="Say how many of each clique type were found in a frame.")
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
            cpar, vpar, args.report)
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
