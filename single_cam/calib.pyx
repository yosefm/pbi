# A walled-off minimal binding of things needed for traditional calibration,
# with as little of the rest dragged in as possible. The goal is to come close
# to making a separate calibration module in the near future.
#
# Epipolar lines are related because they are used to check the calibration on
# single frames, so they are brought in too.

from libc.stdlib cimport malloc, calloc, realloc, free

import numpy as np
cimport numpy as np

from optv.tracking_framebuf cimport TargetArray, target
from optv.calibration cimport Calibration, Exterior, Interior, Glass, ap_52, \
    calibration
from optv.parameters cimport ControlParams, VolumeParams, mm_np, control_par, \
    volume_par
from optv.transforms cimport metric_to_pixel, pixel_to_metric, \
    correct_brown_affin
from optv.vec_utils cimport vec3d

cdef extern from "optv/ray_tracing.h":
    void ray_tracing(double x, double y, calibration* cal, mm_np mm,
        double X[3], double a[3]);
    
cdef extern from "optv/image_processing.h":
    void prepare_image(unsigned char *img, unsigned char *img_hp, 
        int dim_lp, int filter_hp, char *filter_file, control_par *cpar)

cdef extern from "optv/orientation.h":
    ctypedef double vec2d[2]
    ctypedef struct orient_par:
        pass
    enum:
        NPAR
    
    double weighted_dumbbell_precision(vec2d** targets, int num_targs, 
        int num_cams, mm_np *multimed_pars, calibration* cals[], 
        int db_length, double db_weight)
    double point_position(vec2d targets[], int num_cams, mm_np *multimed_pars,
        calibration* cals[], vec3d res);
    int raw_orient(calibration* cal, control_par *cpar, int nfix, vec3d fix[], 
        target pix[]);
    double* orient (calibration* cal_in, control_par *cpar, int nfix, 
        vec3d fix[], target pix[], orient_par *flags, double sigmabeta[20])
    orient_par* read_orient_par(char *filename)
    
cdef extern from "image_processing.h":
    void targ_rec(unsigned char *img0, unsigned char *img, char *par_file,
        int xmin, int xmax, int ymin, int ymax, target *pix, int nr, int* num,
            control_par *cpar)

cdef extern from "optv/multimed.h":
    void move_along_ray(double glob_Z, vec3d vertex, vec3d direct, vec3d out)

cdef extern from "optv/imgcoord.h":
    void img_coord (vec3d pos, calibration *cal, mm_np *mm,
        double *x, double *y)

cdef extern from "optv/epi.h":
    ctypedef struct coord_2d:
        int pnr
        double x, y

cdef extern from "typedefs.h":
    ctypedef struct n_tupel:
        int p[4]

cdef extern from "correspondences.h":
    enum:
        nmax
    void quicksort_target_y(target *pix, int num)
    void quicksort_coord2d_x(coord_2d *crd, int num)
    int correspondences_4 (target pix[][nmax], coord_2d geo[][nmax], int num[],
        volume_par *vpar, control_par *cpar, calibration cals[], n_tupel *con,
        int match_counts[])

cdef extern from "optv/sortgrid.h":
    target* sortgrid(calibration *cal, control_par *cpar, 
        int nfix, vec3d fix[], int num, int eps, target pix[])

def simple_highpass(np.ndarray img, ControlParams cparam):
    cdef np.ndarray hp = np.empty_like(img)
    prepare_image(<unsigned char *>img.data, <unsigned char *>hp.data, 12, 0, 
        NULL, cparam._control_par)
    return hp

ctypedef np.float64_t pos_t

def detect_ref_points(np.ndarray img, int cam, ControlParams cparam, 
    detection_pars="parameters/detect_plate.par"):
    """
    Detects reference points on a calibration image.
    
    Arguments:
    np.ndarray img - a numpy array holding the 8-bit gray image.
    int cam - number of camera that took the picture, needed for getting
        correct parameters for this image.
    ControlParams cparam - an object holding general control parameters.
    
    Returns:
    A TargetArray object holding the targets found.
    """
    cdef:
        TargetArray t = TargetArray()
        target *ret
        target *targs = <target *> calloc(1024*20, sizeof(target))
        np.ndarray img0 = img.copy()
        int num_targs
    
    targ_rec(<unsigned char *>img.data, <unsigned char *>img0.data, detection_pars,
        0, cparam._control_par[0].imx, 1, cparam._control_par[0].imy, targs, 
        cam, &num_targs, cparam._control_par)
    
    ret = <target *>realloc(targs, num_targs * sizeof(target))
    if ret == NULL:
        free(targs)
        raise MemoryError("Failed to reallocate target array.")
    
    t.set(ret, num_targs, 1)
    return t

def external_calibration(Calibration cal, 
    np.ndarray[ndim=2, dtype=pos_t] ref_pts, 
    np.ndarray[ndim=2, dtype=pos_t] img_pts,
    ControlParams cparam):
    """
    Update the external calibration with results of raw orientation, i.e.
    the iterative process that adjust the initial guess' external parameters
    (position and angle of cameras) without internal or distortions.
    
    Arguments:
    Calibration cal - position and other parameters of the camera.
    np.ndarray[ndim=2, dtype=pos_t] ref_pts - an (n,3) array, the 3D known 
        positions of the select 2D points found on the image.
    np.ndarray[ndim=2, dtype=pos_t] img_pts - a selection of pixel coordinates
        of image points whose 3D position is known.
    ControlParams cparam - an object holding general control parameters.
    
    Returns:
    True if iteration succeeded, false otherwise.
    """
    cdef:
        target *targs
        vec3d *ref_coord
    
    ref_pts = np.ascontiguousarray(ref_pts)
    ref_coord = <vec3d *>ref_pts.data
    
    # Convert pixel coords to metric coords:
    targs = <target *>calloc(len(img_pts), sizeof(target))
    
    for ptx, pt in enumerate(img_pts):
        targs[ptx].x = pt[0]
        targs[ptx].y = pt[1]
    
    success = raw_orient (cal._calibration, cparam._control_par, 
        len(ref_pts), ref_coord, targs)
    
    free(targs);
    
    return (True if success else False)

def match_detection_to_ref(Calibration cal,
    np.ndarray[ndim=2, dtype=pos_t] ref_pts, TargetArray img_pts,
    ControlParams cparam, eps=25):
    """
    Creates a TargetArray where the targets are those for which a point in the
    projected reference is close enough to be considered a match, ordered by 
    the order of corresponding references, with "empty targets" for detection 
    points that have no match.
    
    Each target's pnr attribute is set to the index of the target in the array, 
    which is also the index of the associated reference point in ref_pts. 
    
    Arguments:
    Calibration cal - position and other parameters of the camera.
    np.ndarray[ndim=2, dtype=pos_t] ref_pts - an (n,3) array, the 3D known 
        positions of the select 2D points found on the image.
    TargetArray img_pts - detected points to match to known 3D positions.
        Modified inplace.
    ControlParams cparam - an object holding general control parameters.
    int eps - pixel radius of neighbourhood around detection to search for
        closest projection.
    
    Returns:
    TargetArray holding the sorted targets.
    """
    cdef:
        vec3d *ref_coord
        target *sorted_targs
        TargetArray t = TargetArray()
    
    ref_pts = np.ascontiguousarray(ref_pts)
    ref_coord = <vec3d *>ref_pts.data
    
    sorted_targs = sortgrid(cal._calibration, cparam._control_par, 
        len(ref_pts), ref_coord, len(img_pts), eps, img_pts._tarr)
    
    t.set(sorted_targs, len(ref_pts), 1)
    return t

def full_calibration(Calibration cal,
    np.ndarray[ndim=2, dtype=pos_t] ref_pts, TargetArray img_pts,
    ControlParams cparam):
    """
    Performs a full calibration, affecting all calibration structs.
    
    Arguments:
    Calibration cal - current position and other parameters of the camera. Will
        be overwritten with new calibration if iteration succeeded, otherwise 
        remains untouched.
    np.ndarray[ndim=2, dtype=pos_t] ref_pts - an (n,3) array, the 3D known 
        positions of the select 2D points found on the image.
    TargetArray img_pts - detected points to match to known 3D positions.
        Must be sorted by matching ref point (as done by function
        ``match_detection_to_ref()``.
    ControlParams cparam - an object holding general control parameters.
    
    Returns:
    ret - (r,2) array, the residuals in the x and y direction for r points used
        in orientation.
    used - r-length array, indices into target array of targets used.
    err_est - error estimation per calibration DOF. We 
    
    Raises:
    ValueError if iteration did not converge.
    """
    cdef:
        vec3d *ref_coord
        np.ndarray[ndim=2, dtype=pos_t] ret
        np.ndarray[ndim=1, dtype=np.int_t] used
        np.ndarray[ndim=1, dtype=pos_t] err_est
        orient_par *orip
        double *residuals
    
    ref_pts = np.ascontiguousarray(ref_pts)
    ref_coord = <vec3d *>ref_pts.data
    orip = read_orient_par("parameters/orient.par")
    
    err_est = np.empty((NPAR + 1) * sizeof(double))
    residuals = orient(cal._calibration, cparam._control_par, len(ref_pts), 
        ref_coord, img_pts._tarr, orip, <double *>err_est.data)
    
    free(orip)
    
    if residuals == NULL:
        free(residuals)
        raise ValueError("Orientation iteration failed, need better setup.")
    
    ret = np.empty((len(img_pts), 2))
    used = np.empty(len(img_pts), dtype=np.int_)
    
    for ix in range(len(img_pts)):
        ret[ix] = (residuals[2*ix], residuals[2*ix + 1])
        used[ix] = img_pts[ix].pnr()
    
    free(residuals)
    return ret, used, err_est

def epipolar_curve(np.ndarray[ndim=1, dtype=pos_t] image_point,
    Calibration origin_cam, Calibration project_cam, int num_points,
    ControlParams cparam, VolumeParams vparam):
    """
    Get the points lying on the epipolar line from one camera to the other, on
    the edges of the observed volume. Gives pixel coordinates.
    
    Assumes the same volume applies to all cameras.
    
    Arguments:
    np.ndarray[ndim=1, dtype=pos_t] image_point - the 2D point on the image
        plane of the camera seeing the point. Distorted pixel coordinates.
    Calibration origin_cam - current position and other parameters of the 
        camera seeing the point.
    Calibration project_cam - current position and other parameters of the 
        cameraon which the line is projected.
    int num_points - the number of points to generate along the line. Minimum
        is 2 for both endpoints.
    ControlParams cparam - an object holding general control parameters.
    VolumeParams vparam - an object holding observed volume size parameters.
    
    Returns:
    line_points - (num_points,2) array with projection camera image coordinates
        of points lying on the ray stretching from the minimal Z coordinate of 
        the observed volume to the maximal Z thereof, and connecting the camera 
        with the image point on the origin camera.
    """
    cdef:
        np.ndarray[ndim=2, dtype=pos_t] line_points = np.empty((num_points, 2))
        vec3d vertex, direct, pos
        int pt_ix
        double Z
        double *x, *y
        double img_pt[2]
    
    # Move from distorted pixel coordinates to straight metric coordinates.
    pixel_to_metric(img_pt, img_pt + 1, image_point[0], image_point[1], 
        cparam._control_par)
    img_pt[0] -= origin_cam._calibration.int_par.xh
    img_pt[1] -= origin_cam._calibration.int_par.yh
    correct_brown_affin (img_pt[0], img_pt[1], 
        origin_cam._calibration.added_par, img_pt, img_pt + 1)
    
    ray_tracing(img_pt[0], img_pt[1], origin_cam._calibration,
        cparam._control_par.mm[0], vertex, direct)
    
    for pt_ix, Z in enumerate(np.linspace(vparam._volume_par.Zmin_lay[0], 
        vparam._volume_par.Zmax_lay[0], num_points)):
        
        x = <double *>np.PyArray_GETPTR2(line_points, pt_ix, 0)
        y = <double *>np.PyArray_GETPTR2(line_points, pt_ix, 1)
        
        move_along_ray(Z, vertex, direct, pos)
        img_coord(pos, project_cam._calibration, cparam._control_par.mm, x, y)
        metric_to_pixel(x, y, x[0], y[0], cparam._control_par)
    
    return line_points

cdef calibration** cal_list2arr(list cals):
    """
    Allocate a C array with C calibration objects based on a Python list with
    Python Calibration objects.
    """
    cdef:
        calibration **calib
        int num_cals = len(cals)
    
    calib = <calibration **>calloc(num_cals, sizeof(calibration *))
    for cal in range(num_cals):
        calib[cal] = (<Calibration>cals[cal])._calibration
    
    return calib
    
def dumbbell_target_func(np.ndarray[ndim=3, dtype=pos_t] targets, 
    ControlParams cparam, cals, db_length, db_weight):
    """
    Wrap the epipolar convergence test.
    
    Arguments:
    np.ndarray[ndim=3, dtype=pos_t] targets - (num_targets, num_cams, 2) array,
        containing the metric coordinates of each target on the image plane of
        each camera. Cameras must be in the same order for all targets.
    ControlParams cparam - needed for the parameters of the tank through which
        we see the targets.
    cals - a sequence of Calibration objects for each of the cameras, in the 
        camera order of ``targets``.
    db_length - distance between two dumbbell targets.
    db_weight - weight of relative dumbbell size error in target function.
    """
    cdef:
        np.ndarray[ndim=2, dtype=pos_t] targ
        vec2d **ctargets
        calibration **calib = cal_list2arr(cals)
        int cam, num_cams
    
    num_cams = targets.shape[1]
    num_pts = targets.shape[0]
    ctargets = <vec2d **>calloc(num_pts, sizeof(vec2d*))
    
    for pt in range(num_pts):
        targ = targets[pt]
        ctargets[pt] = <vec2d *>(targ.data)
    
    return weighted_dumbbell_precision(ctargets, num_pts, num_cams, 
        cparam._control_par.mm, calib,  db_length, db_weight)

def point_positions(np.ndarray[ndim=3, dtype=pos_t] targets, 
    ControlParams cparam, cals):
    """
    Calculate the 3D positions of the points given by their 2D projections.
    
    Arguments:
    np.ndarray[ndim=3, dtype=pos_t] targets - (num_targets, num_cams, 2) array,
        containing the metric coordinates of each target on the image plane of
        each camera. Cameras must be in the same order for all targets.
    ControlParams cparam - needed for the parameters of the tank through which
        we see the targets.
    cals - a sequence of Calibration objects for each of the cameras, in the 
        camera order of ``targets``.
    
    Returns:
    res - (n,3) array for n points represented by their targets.
    rcm - n-length array, the Ray Convergence Measure for eachpoint.
    """
    cdef:
        np.ndarray[ndim=2, dtype=pos_t] res
        np.ndarray[ndim=1, dtype=pos_t] rcm
        np.ndarray[ndim=2, dtype=pos_t] targ
        calibration **calib = cal_list2arr(cals)
        int cam, num_cams
    
    # So we can address targets.data directly instead of get_ptr stuff:
    targets = np.ascontiguousarray(targets) 
    
    num_cams = targets.shape[1]
    num_pts = targets.shape[0]
    res = np.empty((num_pts,3))
    rcm = np.empty(num_pts)
    
    for pt in range(num_pts):
        targ = targets[pt]
        rcm[pt] = point_position(<vec2d *>(targ.data), num_cams, 
            cparam._control_par.mm, calib, 
            <vec3d>np.PyArray_GETPTR2(res, pt, 0))
    
    return res, rcm

ctypedef target pix_buf[][nmax]
ctypedef coord_2d geo_buf[][nmax]

def correspondences(list img_pts, list cals, VolumeParams vparam,
    ControlParams cparam):
    """
    Get the correspondences for each clique size. Don't care about 
    their actual value for now.
    
    Arguments:
    cals - a list of Calibration objects, each for the camera taking one image.
    img_pts - a list of c := len(cals), containing TargetArray objects, each 
        with the target coordinates of n detections in the respective image.
        The target arrays are clobbered: returned arrays are in y order and
        with the tnr property set.
    VolumeParams vparam - an object holding observed volume size parameters.
    ControlParams cparam - an object holding general control parameters.
    
    Returns:
    sorted_pos - a tuple of (c,?,2) arrays, each with the positions in each of 
        c image planes of points belonging to quadruplets, triplets, pairs 
        found.
    sorted_corresp - a tuple of (c,?) arrays, each with the point identifiers
        of targets belobging to a quad/trip/etc per camera.
    num_targs - total number of targets (must be greater than the sum of 
        previous 3).
    """
    cdef:
        int pt, cam
        int num_cams = len(cals)
        double x, y
        int match
        int *num = <int *> malloc(len(cals) * sizeof(int))
        
        calibration *calib = <calibration *> malloc(
            len(cals) * sizeof(calibration))
        TargetArray targ
        
        target *pix = <target *> malloc(num_cams*nmax * sizeof(target))
        target *curr_pix
        
        coord_2d *geo = <coord_2d *> malloc(num_cams*nmax * sizeof(coord_2d))
        coord_2d *curr_geo
        
        # Return buffers:
        int *match_counts = <int *> malloc(num_cams * sizeof(int))
        n_tupel *corresp_buf = <n_tupel *> malloc(nmax * sizeof(n_tupel))
    
    # Move targets to a C array and create the flat-camera version expected
    # by correspondences_4.
    for cam in range(num_cams):
        calib[cam] = (<Calibration>cals[cam])._calibration[0]
        targ = img_pts[cam]
        num[cam] = len(targ)
        
        quicksort_target_y(targ._tarr, num[cam])
        
        curr_pix = &(pix[cam*nmax])
        curr_geo = &(geo[cam*nmax])
        
        for pt in range(len(targ)):
            curr_pix[0] = targ._tarr[pt]
            curr_pix[0].pnr = pt
            curr_geo[0].pnr = pt
            
            # Flat image coordinates:
            pixel_to_metric(&x, &y, curr_pix.x, curr_pix.y, 
                cparam._control_par);
            x -= calib[cam].int_par.xh
            y -= calib[cam].int_par.yh
            correct_brown_affin (x, y, calib[cam].added_par,
                &(curr_geo[0].x), &(curr_geo[0].y));
            
            curr_pix = &(curr_pix[1])
            curr_geo = &(curr_geo[1])
        
        quicksort_coord2d_x (&(geo[cam*nmax]), len(targ))
    
    # The biz:
    match = correspondences_4 (<pix_buf>pix, <geo_buf>geo, num,
        vparam._volume_par, cparam._control_par, calib, 
        corresp_buf, match_counts)
    
    # Distribute data to return structures:
    sorted_pos = [None]*(num_cams - 1)
    sorted_corresp = [None]*(num_cams - 1)
    last_count = 0
    
    for clique_type in xrange(num_cams - 1):
        num_points = match_counts[clique_type]
        clique_targs = np.full((num_cams, num_points, 2), -999, 
            dtype=np.float64)
        clique_ids = np.full((num_cams, num_points), -1, dtype=np.int_)
        
        # Trace back the pixel target properties through the flat metric
        # intermediary that's x-sorted.
        for cam in range(num_cams):            
            for pt in range(num_points):
                geo_id = corresp_buf[pt + last_count].p[cam]
                if geo_id < 0:
                    continue
                
                p1 = geo[cam*nmax + geo_id].pnr
                clique_ids[cam, pt] = p1

                if p1 > -1:
                    clique_targs[cam, pt, 0] = pix[cam*nmax + p1].x
                    clique_targs[cam, pt, 1] = pix[cam*nmax + p1].y
        
        last_count += num_points
        sorted_pos[clique_type] = clique_targs
        sorted_corresp[clique_type] = clique_ids
    
    # Clean up: copy back modified tergets, and free memory.
    for cam in range(num_cams):
        targ = img_pts[cam]
        curr_pix = &(pix[cam*nmax])
        for pt in range(len(targ)):
            targ._tarr[pt] = curr_pix[pt]
    
    num_targs = match_counts[num_cams - 1]
    free(calib)
    free(pix)
    free(geo)
    free(match_counts)
    free(corresp_buf) # Note this for future returning of correspondences.
    
    return sorted_pos, sorted_corresp, num_targs
