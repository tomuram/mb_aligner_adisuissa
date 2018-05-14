import  numpy as np
cimport numpy as np
import cython
from libc.stdlib cimport rand, RAND_MAX
from libc.limits cimport INT_MAX
from libc.math cimport sqrt, floor, cos, sin, asin, INFINITY, fabs
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libcpp.set cimport set as cpp_set
from cython.operator cimport dereference as deref

## cython: profile=True
## cython: linetrace=True

DEF MinMatchesNumRigid = 2
DEF EPS = 0.000001

cdef enum RansacResultType:
    RANSAC_SUCCESS = 0,
    RANSAC_NOT_ENOUGH_POINTS = -1,
    RANSAC_NO_GOOD_MODEL_FOUND = -2

cdef inline void index1d_to_index2d(size_t n, size_t idx_1d, size_t *out_row, size_t *out_col) nogil:
    """
    Converts a number idx in the range [0, n*(n-1)/2) to a tuple:
    (row, col) of an upper triangular matrix of shape (n*n)
    """
    out_row[0] = n - 2 - <size_t>(floor(sqrt(-8 * idx_1d + 4 * n * (n - 1) - 7)/2.0 - 0.5))
    out_col[0] = idx_1d + 1 + <size_t>(0.5*out_row[0]*out_row[0] - 0.5*(2*n - 3)*out_row[0])

cdef inline int fit_rigid(float p1_x, float p1_y, float q1_x, float q1_y,
                    float p2_x, float p2_y, float q2_x, float q2_y,
                    float *angle, float *t_x, float *t_y) nogil:
    """
    Given 2 matches of points (p1 -> q1, p2 -> q2), returns a tuple:
    (ret_status, angle, t_x, t_y)
    where only if ret_status is 1, angle, t_x, and t_y values are a rigid transform between the
    pair of matches.
    """
    cdef float dx_p, dy_p, sin_angle, cos_angle

    dx_p = p1_x - p2_x
    dy_p = p1_y - p2_y
    if fabs(dx_p) <= EPS or fabs(dy_p) <= EPS:
        #return 0, 0, 0, 0
        return 0

    sin_angle = ((q1_y - q2_y) * dx_p - (q1_x - q2_x) * dy_p) / (dx_p*dx_p + dy_p*dy_p)
    angle[0] = asin(sin_angle)
    cos_angle = cos(angle[0])
    t_x[0] = q1_x - p1_x * cos_angle + p1_y * sin_angle
    t_y[0] = q1_y - p1_x * sin_angle - p1_y * cos_angle
    return 1

cdef inline float compute_rigid_model_score(
                    np.float32_t* X,
                    np.float32_t* y,
                    size_t matches_num,
                    #np.ndarray[np.float32_t, ndim=2, mode='c'] X,
                    #np.ndarray[np.float32_t, ndim=2, mode='c'] y,
                    float angle, float t_x, float t_y,
                    float epsilon,
                    float min_inlier_ratio,
                    float min_num_inlier,
                    np.float32_t *dists2_temp
            ) nogil:
    """
    Applies the rigid transformation for points in X and computes the L2 distance to the points in Y.
    Accepts as inlier each match that has distance at most epsilon.
    """
    cdef int inliers_num = 0
    cdef size_t p_idx
    cdef float new_x, d_x, new_y, d_y, dist2, cos_angle, sin_angle
    cdef float epsilon2 = epsilon * epsilon # epsilon^2, to avoid sqrt later on
    cdef size_t x_idx, y_idx
    #cdef np.ndarray[np.float32_t, ndim=2, mode='c'] X2 = np.empty_like(X)
    #cdef np.float32_t* new_xs = <float *>malloc(matches_num * sizeof(np.float32_t))
    #cdef np.float32_t* new_ys = <float *>malloc(matches_num * sizeof(np.float32_t))
    #cdef np.float32_t* dists2 = <float *>malloc(matches_num * sizeof(np.float32_t))

    # compute the transformed X
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    # Transform each point in X, and compute the L2 distance 
    for p_idx in range(matches_num):
        x_idx = 2 * p_idx
        y_idx = x_idx + 1
        new_x = X[x_idx] * cos_angle - X[y_idx] * sin_angle + t_x
        new_y = X[x_idx] * sin_angle + X[y_idx] * cos_angle + t_y

        d_x = new_x - y[x_idx]
        d_y = new_y - y[y_idx]

        dists2_temp[p_idx] = d_x*d_x + d_y*d_y

    for p_idx in range(matches_num):
        if dists2_temp[p_idx] < epsilon2:
            inliers_num += 1

    #free(new_ys)
    #free(new_xs)
    #free(dists2)
    cdef float accepted_ratio = float(inliers_num) / matches_num
    if inliers_num < min_num_inlier or accepted_ratio < min_inlier_ratio:
        return -1
    return accepted_ratio

cdef void get_rigid_model_inliers(
                    np.float32_t* X,
                    np.float32_t* y,
                    np.int_t* out_inliers,
                    int matches_num,
                    #np.ndarray[np.float32_t, ndim=2, mode='c'] X,
                    #np.ndarray[np.float32_t, ndim=2, mode='c'] y,
                    #np.ndarray[np.int_t, ndim=1, mode='c'] out_inliers,
                    float angle, float t_x, float t_y,
                    float epsilon
            ) nogil:
    """
    Applies the rigid transformation for points in X and computes the L2 distance to the points in Y.
    Accepts as inlier each match that has distance at most epsilon.
    Updates the out_inliers array to have 1 for inliers and 0 for outliers.
    """
    cdef int p_idx
    cdef float new_x, d_x, new_y, d_y, dist2, cos_angle, sin_angle
    cdef float epsilon2 = epsilon * epsilon # epsilon^2, to avoid sqrt later on

    # compute the transformed X
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    # Transform each point in X, and compute the L2 distance 
    for p_idx in range(matches_num):
        new_x = X[2 * p_idx] * cos_angle - X[2 * p_idx + 1] * sin_angle + t_x
        d_x = new_x - y[2 * p_idx]
        new_y = X[2 * p_idx] * sin_angle + X[2 * p_idx + 1] * cos_angle + t_y
        d_y = new_y - y[2 * p_idx + 1]

        dist2 = d_x*d_x + d_y*d_y
        if dist2 < epsilon2:
            out_inliers[p_idx] = 1
        else:
            out_inliers[p_idx] = 0

cdef void random_choice_no_repeat(
            size_t n,
            size_t c,
            np.ndarray[np.int32_t, ndim=1, mode='c'] out_choices
            ):
    """
    Chooses c numbers out of the set [0, n), without repetitions
    """
    cdef int insert_counter = 0
    cdef cpp_set[int] chosen_set
    cdef int rand_num
    cdef np.int32_t* out_choices_arr = &out_choices[0]

    if c >= n/2:
        # If we need to choose many numbers, just use numpy's random
        out_choices[:] = np.random.choice(n, c, False)
    else:
        # try a faster pseudo random way to generate the numbers
        # TODO - try to find a better method for no repetition is w.h.p, but that might not be guaranteed (similar to linear feedback shift register)

        with nogil:
            while insert_counter < c:            
                # draw a random number and make sure it wasn't used
                rand_num = rand() % n
                while chosen_set.find(rand_num) != chosen_set.end():
                    rand_num = int(n * rand() / RAND_MAX)

                out_choices_arr[insert_counter] = rand_num
                chosen_set.insert(rand_num)
                insert_counter += 1



##@cython.profile(True)
##@cython.binding(True)
##@cython.linetrace(True)
@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices ([-1,-1])
def ransac_rigid(
            #np.ndarray[np.float32_t, ndim=3, mode='c'] sample_matches_T,
            #np.ndarray[np.float32_t, ndim=3, mode='c'] test_matches_T,
            sample_matches,
            test_matches,
            int iterations,
            float epsilon,
            float min_inlier_ratio,
            float min_num_inlier,
            float max_rot_deg
        ):
    """
    Ransac optimized for 2d rigid transformations only
    """
    printf("len(sample_matches[0]): %d\n", len(sample_matches[0]))
    if len(sample_matches[0]) < MinMatchesNumRigid:
        return RANSAC_NOT_ENOUGH_POINTS, None, None, None

    cdef float best_model_score = 0
    cdef float best_model_angle
    cdef float best_model_t_x
    cdef float best_model_t_y
    cdef int len_sample_matches0 = len(sample_matches[0])
    cdef int len_test_matches0 = len(test_matches[0])
    # Avoiding repeated indices permutations using a dictionary
    # Limit the number of possible matches that we can search for using n choose k
    cdef long max_combinations_real = long(long(len_sample_matches0) * (len_sample_matches0 - 1) / 2) # N choose 2
    cdef int max_combinations = int(min(max_combinations_real, INT_MAX - 1))
    cdef int max_iterations = min(iterations, max_combinations)
    cdef size_t idx_1d, pq1_idx, pq2_idx
    cdef int fit_res
    cdef float model_angle, model_t_x, model_t_y
    cdef float proposed_model_score
    # choose max_iterations different pairs of matches to create the transformation
    # Note that we'll randomly choose a number between 0 to max_combinations-1, and then convert it
    # to a single pair of matches (see: https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix)
    #printf("max_combs: %d, max_iterations: %d\n", max_combinations, max_iterations)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] choices_1d_idxs = \
            np.empty(max_iterations, dtype=np.int32)
            #np.random.choice(max_combinations, max_iterations, False)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] sample_matches0_arr = np.ascontiguousarray(sample_matches[0])
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] sample_matches1_arr = np.ascontiguousarray(sample_matches[1])
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] test_matches0_arr = np.ascontiguousarray(test_matches[0])
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] test_matches1_arr =np.ascontiguousarray( test_matches[1])
    cdef np.float32_t *sample_matches0 = &sample_matches0_arr[0, 0]
    cdef np.float32_t *sample_matches1 = &sample_matches1_arr[0, 0]
    cdef np.float32_t* dists2_temp = <float *>malloc(len_test_matches0 * sizeof(np.float32_t))


    random_choice_no_repeat(max_combinations, max_iterations, choices_1d_idxs)

    # with nogil:
    for i in range(max_iterations):
        idx_1d = choices_1d_idxs[i]
        index1d_to_index2d(len_sample_matches0, idx_1d, &pq1_idx, &pq2_idx)
        fit_res = fit_rigid(
            sample_matches0[2 * pq1_idx], sample_matches0[2 * pq1_idx + 1], sample_matches1[2 * pq1_idx], sample_matches1[2 * pq1_idx + 1], # p1_x, p1_y, q1_x, q1_y
            sample_matches0[2 * pq2_idx], sample_matches0[2 * pq2_idx + 1], sample_matches1[2 * pq2_idx], sample_matches1[2 * pq2_idx + 1], # p2_x, p2_y, q2_x, q2_y
            #sample_matches_T0[pq1_idx], sample_matches_T0[pq1_idx + len_sample_matches0], sample_matches_T1[pq1_idx], sample_matches_T1[pq1_idx + len_sample_matches0], # p1_x, p1_y, q1_x, q1_y
            #sample_matches_T0[pq2_idx], sample_matches_T0[pq2_idx + len_sample_matches0], sample_matches_T1[pq2_idx], sample_matches_T1[pq2_idx + len_sample_matches0], # p2_x, p2_y, q2_x, q2_y
            &model_angle, &model_t_x, &model_t_y
            )
        if fit_res == 0:
            continue

        # compute the model's score (on the test_matches)
        proposed_model_score = compute_rigid_model_score(&test_matches0_arr[0, 0], &test_matches1_arr[0, 0], len_test_matches0,
            model_angle, model_t_x, model_t_y, epsilon, min_inlier_ratio, min_num_inlier, dists2_temp)

        if proposed_model_score > best_model_score:
            best_model_score = proposed_model_score
            best_model_angle = model_angle
            best_model_t_x = model_t_x
            best_model_t_y = model_t_y

    free(dists2_temp)

    if best_model_score == 0:
        # No good model found
        return None, None

    return RANSAC_SUCCESS, (best_model_angle, best_model_t_x, best_model_t_y)#, get_rigid_model_inliers output
        

def ransac(
            np.ndarray[np.float32_t, ndim=3, mode='c'] sample_matches,
            np.ndarray[np.float32_t, ndim=3, mode='c'] test_matches,
            int target_model_type,
            int iterations,
            float epsilon,
            float min_inlier_ratio,
            float min_num_inlier,
            float det_delta,
            float max_stretch,
            float max_rot_deg,
            float tri_angles_comparator
        ):
    """
    target_model_type: 1 - Rigid, 3 - Affine
    """
    assert(len(sample_matches[0]) == len(sample_matches[1]))
    assert(target_model_type == 1 or target_model_type == 3)

#     return ransac_rigid(sample_matches.data, len(sample_matches[0]),
#                         test_matches.data, len(test_matches[0]),
#                         iterations,
#                         epsilon,
#                         min_inlier_ratio,
#                         min_num_inlier,
#                         max_rot_deg
#                        )

#     best_model = None
#     best_model_score = 0 # The higher the better
#     best_inlier_mask = None
#     best_model_mean_dists = 0
#     proposed_model = Transforms.create(target_model_type)


#     max_rot_deg_cos = None
#     if max_rot_deg is not None:
#         max_rot_deg_cos = math.cos(max_rot_deg * math.pi / 180.0)
#         #print("max_rot_deg: {},  max_rot_deg_cos: {}, {}".format(max_rot_deg, max_rot_deg_cos, max_rot_deg * math.pi / 180.0))
# 
#     if proposed_model.MIN_MATCHES_NUM > sample_matches[0].shape[0]:
#         logger.report_event("RANSAC cannot find a good model because the number of initial matches ({}) is too small.".format(sample_matches[0].shape[0]), log_level=logging.WARN)
#         return None, None, None


