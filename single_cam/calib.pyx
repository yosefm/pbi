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

cdef extern from "optv/trafo.h":
    void metric_to_pixel(double *x_pixel, double* y_pixel,
        double x_metric, double y_metric, control_par* parameters)
    void pixel_to_metric(double *x_metric, double *y_metric,
        double x_pixel, double y_pixel, control_par* parameters)
    void correct_brown_affin (double x, double y, ap_52 ap,
        double *x1, double *y1)

cdef extern from "optv/ray_tracing.h":
    void ray_tracing(double x, double y, calibration* cal, mm_np mm,
        double X[3], double a[3]);
    
cdef extern from "optv/image_processing.h":
    void prepare_image(unsigned char *img, unsigned char *img_hp, 
        int dim_lp, int filter_hp, char *filter_file, control_par *cpar)

cdef extern from "optv/vec_utils.h":
    ctypedef double vec3d[3]

cdef extern from "optv/orientation.h":
    ctypedef double vec2d[2]
    double weighted_dumbbell_precision(vec2d** targets, int num_targs, 
        int num_cams, mm_np *multimed_pars, calibration* cals[], 
        int db_length, double db_weight)
    double point_position(vec2d targets[], int num_cams, mm_np *multimed_pars,
        calibration* cals[], vec3d res);

cdef extern from "image_processing.h":
    void targ_rec(char *img0, char *img, char *par_file,
        int xmin, int xmax, int ymin, int ymax, target *pix, int nr, int* num,
            control_par *cpar)

cdef extern from "optv/imgcoord.h":
    void img_coord (vec3d pos, calibration *cal, mm_np *mm,
        double *x, double *y)

cdef extern from "optv/multimed.h":
    void move_along_ray(double glob_Z, vec3d vertex, vec3d direct, vec3d out)
    
cdef extern from "optv/epi.h":
    ctypedef struct coord_2d:
        int pnr
        double x, y

cdef extern from "typedefs.h":
    ctypedef struct coord_3d:
        int pnr
        double x, y, z
    
    ctypedef struct n_tupel:
        pass

cdef extern from "correspondences.h":
    enum:
        nmax
    void quicksort_coord2d_x (coord_2d *crd, int num)
    int correspondences_4 (target pix[][nmax], coord_2d geo[][nmax], int num[],
        volume_par *vpar, control_par *cpar, calibration cals[], n_tupel *con,
        int match_counts[])

cdef extern from "orientation.h":
    int orient_v3 (calibration init_cal, control_par *cpar,
        int nfix, coord_3d fix[], coord_2d crd[], calibration *res_cal,
        int nr, double resid_x[], double resid_y[], int pixnr[], int *num_used)
    int raw_orient_v3 (calibration init_cal, control_par *cpar,
        int nfix, coord_3d fix[], coord_2d crd[], calibration *res_cal,
        int nr, int only_show)

cdef public coord_2d crd[4][nmax] # Temporary, needed by still-unused functions in orientation.c
cdef public target pix[4][nmax] # ditto
cdef public int nfix # ditto
cdef public int ncal_points[4] # Same for sortgrid.c
cdef public int x_calib[4][1000] # ditto
cdef public int y_calib[4][1000] # ditto
cdef public int z_calib[4][1000] # ditto

cdef extern void sortgrid_man(calibration *cal, int nfix, coord_3d fix[], 
    int num, target pix[], int n_img, control_par *cpar)

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
    
    targ_rec(<char *>img.data, <char *>img0.data, detection_pars,
        0, cparam._control_par[0].imx, 1, cparam._control_par[0].imy, targs, 
        cam, &num_targs, cparam._control_par)
    
    ret = <target *>realloc(targs, num_targs * sizeof(target))
    if ret == NULL:
        free(targs)
        raise MemoryError("Failed to reallocate target array.")

    t.set(ret, num_targs, 1)
    return t

def pixel_2D_coords(Calibration cal, np.ndarray[ndim=2, dtype=pos_t] pos3d,
    ControlParams cparam):
    """
    Translate an array of 3D coordinates into an array of the corresponding 2D
    coordinates in the image plane of the given camera.
    
    Arguments:
    Calibration cal - position and other parameters of the camera.
    np.ndarray pos3d - (n,3) array, the 3D points to project (metric, usually
        mm but depends on parameters).
    ControlParams cparam - an object holding general control parameters.
    
    Returns:
    pos2d - (n,2) array, the corresponding pixel coordinates on the image 
        plane.
    """
    cdef:
        double *xp, *yp
        np.ndarray[ndim=2, dtype=pos_t] pos2d = np.empty(
            (pos3d.shape[0], 2))
        np.ndarray[ndim=1, dtype=pos_t] point
    
    for pn, point in enumerate(pos3d):
        xp = <double *>np.PyArray_GETPTR2(pos2d, pn, 0)
        yp = <double *>np.PyArray_GETPTR2(pos2d, pn, 1)
        
        img_coord (<double *>point.data, cal._calibration, 
            cparam._control_par.mm, xp, yp)
        metric_to_pixel(xp, yp, xp[0], yp[0], cparam._control_par)

    return pos2d

def image_coords_metric(np.ndarray[ndim=2, dtype=pos_t] arr, 
    ControlParams cparam):
    """
    Convert pixel coordinates to metric coordinates on the image plane of a 
    camera. 
    
    Arguments:
    np.ndarray arr - (n,2) array, the 2D points in pixel coordinates.
    ControlParams cparam - an object holding general control parameters.
    """
    cdef:
        np.ndarray[ndim=2, dtype=pos_t] metric = np.empty_like(arr)
        double *xp, *yp
    
    for pn in range(arr.shape[0]):
        xp = <double *>np.PyArray_GETPTR2(metric, pn, 0)
        yp = <double *>np.PyArray_GETPTR2(metric, pn, 1)
        
        pixel_to_metric(xp, yp, arr[pn,0], arr[pn,1], cparam._control_par)
    
    return metric

cdef coord_3d *array2coord_3d(np.ndarray[ndim=2, dtype=pos_t] arr):
    """
    Convert an array of 3D points to an array of coord_3d points. Number the
    resulting coord_3d objects (the pnr attribute) by the index of the point
    in ``arr``.
    
    Arguments:
    np.ndarray[ndim=2, dtype=pos_t] arr - (n,3) array to convert
    
    returns:
    coord3d *ret - pointer to the memory allocated for the converted array.
    """
    cdef coord_3d *ret = <coord_3d *>calloc(len(arr), sizeof(coord_3d))
    
    for ptx, pt in enumerate(arr):
        ret[ptx].x = pt[0]
        ret[ptx].y = pt[1]
        ret[ptx].z = pt[2]
        ret[ptx].pnr = ptx
    
    return ret
    
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
        coord_2d *metric_coord
        coord_3d *ref_coord = array2coord_3d(ref_pts)
    
    # Convert pixel coords to metric coords:
    metric_coord = <coord_2d *>calloc(len(img_pts), sizeof(coord_2d))
    
    for ptx, pt in enumerate(img_pts):
        pixel_to_metric(&(metric_coord[ptx].x), &(metric_coord[ptx].y),
                        pt[0], pt[1], cparam._control_par)
        correct_brown_affin(metric_coord[ptx].x, metric_coord[ptx].y, 
            cal._calibration.added_par, 
            &(metric_coord[ptx].x), &(metric_coord[ptx].y))
    
    success = raw_orient_v3(cal._calibration[0], cparam._control_par, 
        len(ref_pts), ref_coord, metric_coord, cal._calibration, 0, 0)
    
    free(metric_coord);
    free(ref_coord);
    
    return (True if success else False)

def match_detection_to_ref(Calibration cal,
    np.ndarray[ndim=2, dtype=pos_t] ref_pts, TargetArray img_pts,
    ControlParams cparam):
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
    """
    cdef:
        coord_3d *ref_coord = array2coord_3d(ref_pts)
    
    sortgrid_man(cal._calibration, len(ref_pts), ref_coord, len(img_pts), 
        img_pts._tarr, 0, cparam._control_par)
    
    free(ref_coord);

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
    
    Raises:
    ValueError if iteration did not converge.
    """
    cdef:
        coord_2d *metric_coord
        coord_3d *ref_coord = array2coord_3d(ref_pts)
        double resid_x[1000]
        double resid_y[1000]
        int pixnr[1000];
        int num_used
        np.ndarray[ndim=2, dtype=pos_t] ret
        np.ndarray[ndim=1, dtype=np.int_t] used
    
    # Pixel to metric coordinates, with preserved internal numbering.
    metric_coord = <coord_2d *>calloc(len(img_pts), sizeof(coord_2d))
    for ptx, pt in enumerate(img_pts):
        x, y = pt.pos() # Using the Python object here, speed not necessary.
        pixel_to_metric(&(metric_coord[ptx].x), &(metric_coord[ptx].y),
            x, y, cparam._control_par)
        metric_coord[ptx].pnr = pt.pnr()
    
    success = orient_v3(cal._calibration[0], cparam._control_par, len(ref_pts), 
        ref_coord, metric_coord, cal._calibration, 0, resid_x, resid_y, pixnr, 
        &num_used);
    
    free(ref_coord);
    free(metric_coord);
    
    if success == 0:
        raise ValueError("Orientation iteration failed, need better setup.")
    
    ret = np.empty((num_used, 2))
    used = np.empty(num_used, dtype=np.int_)
    for ix in range(num_used):
        ret[ix] = (resid_x[ix], resid_y[ix])
        used[ix] = pixnr[ix]
    
    return ret, used

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
    """
    cdef:
        np.ndarray[ndim=1, dtype=pos_t] res
        np.ndarray[ndim=2, dtype=pos_t] targ
        vec2d **ctargets
        calibration **calib = cal_list2arr(cals)
        int cam, num_cams
    
    num_cams = targets.shape[1]
    num_pts = targets.shape[0]
    ctargets = <vec2d **>calloc(num_pts, sizeof(vec2d*))
    res = np.empty((num_pts,3))
    
    for pt in range(num_pts):
        targ = targets[pt]
        point_position(<vec2d *>(targ.data), num_cams, cparam._control_par.mm,
            calib, <vec3d>np.PyArray_GETPTR2(res, pt, 0))
    
    return res

ctypedef target pix_buf[][nmax]
ctypedef coord_2d geo_buf[][nmax]

def count_correspondences(list img_pts, list cals, VolumeParams vparam,
    ControlParams cparam):
    """
    Get the number of correspondences for each clique size. Don't care about 
    their actual value for now.
    
    Arguments:
    cals - a list of Calibration objects, each for the camera taking one image.
    img_pts - a list of len(cals), containing TargetArray objects, each with 
        the target coordinates of n detections in the respective image.
    VolumeParams vparam - an object holding observed volume size parameters.
    ControlParams cparam - an object holding general control parameters.
    
    Returns:
    a tuple with the number of quadruplets, triplets, pairs found and 
    total number of targets (must be greater than the sum of previous 3).
    """
    cdef:
        double x, y
        int match
        int *num = <int *> malloc(len(cals) * sizeof(int))
        
        calibration *calib = <calibration *> malloc(
            len(cals) * sizeof(calibration))
        TargetArray targ
        
        target *pix = <target *> malloc(len(cals)*nmax * sizeof(target))
        target *curr_pix
        
        coord_2d *geo = <coord_2d *> malloc(len(cals)*nmax * sizeof(coord_2d))
        coord_2d *curr_geo
        
        # Return buffers:
        int *match_counts = <int *> malloc(len(cals) * sizeof(int))
        n_tupel *corresp_buf = <n_tupel *> malloc(nmax * sizeof(n_tupel))
    
    # Move targets to a C array and create the flat-camera version expected
    # by correspondences_4.
    for cam in range(len(cals)):
        calib[cam] =  (<Calibration>cals[cam])._calibration[0]
        targ = img_pts[cam]
        curr_pix = &(pix[cam*nmax])
        curr_geo = &(geo[cam*nmax])
        
        num[cam] = len(targ)
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
    ret = tuple(match_counts[c] for c in xrange(len(cals)))
    
    # Clean up.
    free(calib)
    free(pix)
    free(geo)
    free(match_counts)
    free(corresp_buf) # Note this for future returning of correspondences.
    
    return ret