# -*- coding: utf-8 -*-
"""
Evolutionary solution to the camera position. Based on a correspondence-count
target function instead of detection/projection matching.

Created on Thu Mar 10 10:25:17 2016

@author: yosef
"""

import numpy as np, matplotlib.pyplot as pl, signal
from util.evolution import gen_calib, cauchy_mutation, recombination, choose_breeders
from calib import count_correspondences

from parallel_runner import PoolWorker

wrap_it_up = False
def interrupt(signum, frame):
    global wrap_it_up
    wrap_it_up = True
    
signal.signal(signal.SIGINT, interrupt)

def fitness(solution, detections, glass_vecs, vpar, cpar, 
    weights=np.r_[4, 2, 1]):
    """
    Checks the fitness of an evolutionary solution of calibration values to 
    target points. Fitness is the sum of squares of the distance from each 
    guessed point to the closest neighbor.
    
    Arguments:
    solution - array, flattened rows for each camera. Each row concatenates: 
        position of intersection with Z=0 plane; 3 angles of exterior 
        calibration; primary point (xh,yh,cc); 3 radial distortion parameters; 
        2 decentering parameters.
    detections - a list of TargetArray objects, one per camera.
    glass_vecs - (c,3) array, one glass vector for each of c cameras. This is 
        the only part of the calibration struct that remains constant 
        throughout the iteration.
    vpar - a VolumeParams object with observed volume size and criteria for 
        correspondence. Don't ask me why the forefathers put those in the same 
        place.
    cpar - a ControlParams object with image data.
    weights - array, resp. the weight of a quadruplet, triplet, pair. 
    """
    num_cams = glass_vecs.shape[0]
    solution = solution.reshape(num_cams, -1)
    
    cals = []
    for cam in range(num_cams):
        # Breakdown of of agregate solution vector:
        inters = np.zeros(3)
        inters[:2] = solution[cam, :2]
        R = solution[cam, 2]
        angs = solution[cam, 3:6]
        prim_point = solution[cam, 6:9]
        rad_dist = solution[cam, 9:12]
        decent = solution[cam, 12:14]
            
        # Compare known points' projections to detections:
        cal = gen_calib(inters, R, angs, glass_vecs[cam], prim_point, 
            rad_dist, decent)
        cals.append(cal)
    
    matches = count_correspondences(detections, cals, vpar, cpar)
    return np.sum(weights * np.r_[matches][:3])

class FitnessProc(PoolWorker):
    def __init__(self, tasks, command_pipe, results_queue, # pool comm
        detections, glass_vecs, vpar, cpar, weights=np.r_[4, 2, 1]):
        """
        See fitness() and parallel_runner module.
        """
        PoolWorker.__init__(self, tasks, command_pipe, results_queue)
        
        self._vpar = vpar
        self._cpar = cpar
        self._glass = glass_vecs
        self._weights = weights
        self._detect = detections
    
    def job(self, prm):
        """
        Parameters in ``prm``:
        solution - a solution chromosome as understood by fitness().
        
        Return value is a tuple: (solution, fitness)
        """
        solution = np.r_[prm]
        f = fitness(solution, self._detect, self._glass, self._vpar, self._cpar, 
            self._weights)
        return (prm, f)

def show_current(fits, pos):
    print()
    print("************************")
    print((fits.min(), fits.max()))
    print() 
    print("best solution:")
    print((pop[fits.argmax()]))
    print("************************")
    print()

if __name__ == "__main__":
    import argparse, yaml, time
    import numpy.random as rnd
    
    from optv.parameters import VolumeParams, ControlParams, TargetParams
    from optv.segmentation import target_recognition
    from util.openptv import simple_highpass
    
    from multiprocessing import Pipe, Queue, cpu_count
    from queue import Empty
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', 
        help="A YAML file with calibration and image properties.")
    parser.add_argument('--procs', '-p', default=4, type=int,
        help="Number of parallel processes.")
    args = parser.parse_args()
    
    yaml_args = yaml.load(open(args.config,'r'),yaml.CLoader)
    
    control_args = yaml_args['scene']
    cam_args = yaml_args['cameras']
    control_args['cams'] = len(cam_args)
    cpar = ControlParams(**control_args)
    vpar = VolumeParams()
    vpar.read_volume_par(yaml_args['volume_params'])
    targ_par = TargetParams(**yaml_args['detection'])
    
    images = []
    glass_vecs = []
    bounds = []
    detections = []
    for cix, cam_spec in enumerate(cam_args):
        img = pl.imread(cam_spec['image'])
        hp = simple_highpass(img, cpar)
        targs = target_recognition(hp, targ_par, cix, cpar)
        
        images.append(img)
        glass_vecs.append(np.r_[cam_spec['glass_vec']])
        bounds.extend(cam_spec['bounds'])
        detections.append(targs)
    
    glass_vecs = np.array(glass_vecs)
    bounds = np.array(bounds)
    
    # Parallel processing setup.
    # Maximum queue size mandates blocking until a slot is free
    results = Queue()
    num_procs = args.procs
    tasks = Queue(num_procs*2)
    w = []
    
    for p in range(num_procs):
        pside, cside = Pipe()
        t = FitnessProc(tasks, cside, results, 
            detections, glass_vecs, vpar, cpar)
        w.append((t, pside))
        t.start()

        time.sleep(0.5)
        
    # Start genetic stuff.
    
    # Initial population and fitness:
    pop_size = 200
    num_iters = 1000000
    mutation_chance = 0.01
    pop = np.empty((pop_size, len(bounds)))
    fits = np.empty(pop_size)
    sols_tried = 0
    sols_accepted = 0
    
    while sols_tried < num_iters:
        # Check if Ctrl-C event happened during previous iteration:
        if wrap_it_up:
            break
        
        print(sols_tried)
        if sols_tried < pop_size:
            # feed a random solution to the queue:
            sol = [rnd.rand()*(maxb - minb) + minb for minb, maxb in bounds]
            tasks.put(sol)
            sols_tried += 1
            
        elif sols_accepted > pop_size/2:
            # select breeders.
            breeders = choose_breeders(fits[:sols_accepted])
            sol = recombination(*pop[breeders])
            cauchy_mutation(sol, bounds, mutation_chance, 10)
        
            tasks.put(sol)
            sols_tried += 1

        # Check results:
        try:
            while True:
                sol, f = results.get_nowait()
                
                if sols_accepted < pop_size:
                    pop[sols_accepted] = np.array(sol)
                    fits[sols_accepted] = f
                    
                    sols_accepted += 1
                    print(("accepted", sols_accepted))
                    
                else:
                    # Compete with others for insert:
                    loser = fits.argmin()
                    if f > fits[loser]:
                        pop[loser] = np.array(sol)
                        fits[loser] = f
                
        except Empty:
            pass
        
        # respawn segfaulted children instead of solving the stupid segfault,
        # for now.
        if sols_tried % 20 == 0:
            for p in w:
                if not p[0].is_alive():
                    print(" *** respawn ***")
                    p[0].join()
                    w.remove(p)
                    pside, cside = Pipe()
                    t = FitnessProc(tasks, cside, results, 
                        detections, glass_vecs, vpar, cpar)
                    w.append((t, pside))
                    t.start()
                    time.sleep(0.1)
                    
        if (sols_accepted >= pop_size) and (sols_tried % 100 == 0):
            show_current(fits, pop)
        
        time.sleep(0.005)
    
    show_current(fits, pop)
    print(fits)
            
    for p in w:
        p[0].terminate()
