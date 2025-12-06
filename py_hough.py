import numpy as np

fabs = np.abs
sqrt = np.sqrt
ceil = lambda v : int(np.ceil(v))


def _probabilistic_hough_line(img, threshold,
                              line_length, line_gap,
                              theta,
                              rng=None,
                              verbose=False
                             ):
    """Return lines from a progressive probabilistic line Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int
        Threshold in the accumulator to detect lines against noise.
    line_length : int
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggressively.
    theta : (K,) ndarray of float64
        Angles at which to compute the transform, in radians.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.

    Returns
    -------
    lines : list
        List of lines identified, lines in format ((x0, y0), (x1, y1)),
        indicating line start and end.

    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.

    Notes
    -----

    The algorithm (from [1]_) is the following:

    1. Check the input image, if it is empty then finish.
    2. Update the accumulator with a single pixel randomly selected from the
       input image.
    3. Remove pixel from input image.
    4. Check if the highest peak in the accumulator that was modified by the
       new pixel is higher than threshold. If not then goto 1.
    5. Look along a corridor specified by the peak in the accumulator, and find
       the longest segment of pixels either continuous or exhibiting a gap not
       exceeding a given threshold.
    6. Remove the pixels in the segment from input image.
    7. Unvote from the accumulator all the pixels from the line that have
       previously voted.
    8. If the line segment is longer than the minimum length add it into the
       output list.
    9. goto 1.

    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    height = img.shape[0]
    width = img.shape[1]

    # compute the bins and allocate the accumulator array
    mask = \
        np.zeros((height, width), dtype=np.uint8)
    line_ends = np.zeros((2, 2), dtype=np.intp)
    nlines = 0
    lines_max = 2 ** 15  # maximum line number cutoff.
    lines = np.zeros((lines_max, 2, 2),
                                                dtype=np.intp)
    # Assemble n_rhos by n_thetas accumulator array.
    max_distance = ceil((sqrt(img.shape[0] * img.shape[0] +
                                          img.shape[1] * img.shape[1])))
    accum = np.zeros((max_distance * 2,
                      theta.shape[0]),
                     dtype=np.int64)
    rho_idx_offset = max_distance
    nthetas = theta.shape[0]

    line_pixels = np.zeros((max_distance, 2), dtype=np.intp)

    # compute sine and cosine of angles
    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    # find the nonzero indexes
    y_idxs, x_idxs = np.nonzero(img)

    # mask all non-zero indexes
    mask[y_idxs, x_idxs] = 1

    rng = np.random.default_rng(rng)
    rand_idxs = np.arange(len(x_idxs), dtype=np.intp)
    rng.shuffle(rand_idxs)

    for pt_no, idx in enumerate(rand_idxs):  # Select random non-zero point (step 1 above).
        x = x_idxs[idx]
        y = y_idxs[idx]
        vprint('xy', x, y)

        # Skip if previously eliminated by detection in earlier line
        # search.
        if not mask[y, x]:
            vprint('Mask empty at xy')
            continue

        value = 0
        max_value = 0  # Max value in accumulator, start value.
        max_theta_idx = -1  # Index into {c,s}theta arrays, start value.

        # Apply Hough transform on point (step 2 above).
        for j in range(nthetas):
            rho = ctheta[j] * x + stheta[j] * y
            rho_idx = round(rho) + rho_idx_offset
            accum[rho_idx, j] += 1
            value = accum[rho_idx, j]
            if value > max_value:
                max_value = value
                max_theta_idx = j
                max_rho_idx = rho_idx
        if max_value < threshold:  # Step 4 above.
            vprint('Threshold not passed')
            continue

        if np.sum(accum == max_value) > 1:
            vprint('Multiple maxima in accum')

        lc = ctheta[max_theta_idx]
        ls = stheta[max_theta_idx]
        n_line_pixels = find_line2(x, y, mask, ls, lc, line_ends,
                                   line_pixels, line_gap, vprint)

        # Confirm line length is sufficient.
        x_len = line_ends[1, 0] - line_ends[0, 0]
        y_len = line_ends[1, 1] - line_ends[0, 1]
        LL = sqrt(x_len * x_len + y_len * y_len)
        if not LL  >= line_length:
            vprint(f'Not long enough at {LL}')
            continue

        vprint(f'Clearing line len {LL}')
        vprint("Line points", line_pixels[:n_line_pixels])
        vprint("theta", theta[max_theta_idx])
        # Pass 2: reset accumulator and mask for points on line (steps 6
        # and 7 above).
        for i in range(n_line_pixels):
            x1 = line_pixels[i, 0]
            y1 = line_pixels[i, 1]
            # if not mask[y1, x1]:
            #    continue
            mask[y1, x1] = 0  # Remove point.
            for j in range(nthetas):  # Remove accumulator votes.
                rho = ctheta[j] * x1 + stheta[j] * y1
                rho_idx = round(rho) + rho_idx_offset
                accum[rho_idx, j] -= 1

        # Add line to the result (step 8 above).
        lines[nlines] = line_ends
        nlines += 1
        if nlines >= lines_max:
            break

    return ([((line[0, 0], line[0, 1]), (line[1, 0], line[1, 1]))
            for line in lines[:nlines]],
            max_rho_idx - rho_idx_offset,
            rho_idx_offset,
            theta[max_theta_idx]
           )


def find_line(x, y, mask, ls, lc, line_ends, line_pixels, line_gap,
              vprint):

    height = mask.shape[0]
    width = mask.shape[1]

    # From the random point (x, y), walk in opposite directions and
    # find line beginning and end (step 5 above).
    shift = 16
    # Line equation is r = cos theta x + sin theta y.  Rearranging:
    # y = r / sin theta  - cos theta x / sin theta, and slope
    # is -cos theta / sin theta .
    mls = -ls
    slope = mls / lc
    xflag = fabs(slope) > 1
    x0 = x  # Coordinate shift 16 if abs(slope) > 1 False else x
    y0 = y  # Coordinate shift 16 if abs(slope) > 1 True else y
    # calculate gradient of walks using fixed point math
    if xflag:  # abs(y) increases faster than abs(x).
        if mls > 0:
            dx0 = 1
        else:
            dx0 = -1
        # y0, dy0 shifted.  Push value into upper bits.
        dy0 = round(lc * (1 << shift) / fabs(mls))
        y0 = (y0 << shift) + (1 << (shift - 1))
        vprint('Slope, dx, dy', slope, dx0, dy0 / d)
    else:  # abs(x) increases faster than abs(x).
        if lc > 0:
            dy0 = 1
        else:
            dy0 = -1
        # x0, dx0 shifted.  Push value into upper bits.
        dx0 = round(mls * (1 << shift) / fabs(lc))
        x0 = (x0 << shift) + (1 << (shift - 1))
        vprint('Slope, dx, dy', slope, dx0 / d, dy0)
    # pass 1: walk the line, merging lines less than specified gap
    # length (step 5 continued).
    n_line_pixels = 0
    for k in range(2):
        gap = 0
        px = x0
        py = y0
        dx = dx0
        dy = dy0
        if k > 0:  # Walk in opposite direction.
            dx = -dx
            dy = -dy
        while True:
            if xflag:
                x1 = px
                y1 = py >> shift  # Pull value from upper bits.
            else:
                x1 = px >> shift
                y1 = py
            # check when line exits image boundary
            if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
                break
            gap += 1
            if mask[y1, x1]:  # Hit remaining pixel, continue line.
                gap = 0
                line_ends[k, 0] = x1
                line_ends[k, 1] = y1
                # Record presence of in-mask pixel on line.
                line_pixels[n_line_pixels, 0] = x1
                line_pixels[n_line_pixels, 1] = y1
                n_line_pixels += 1
            elif gap > line_gap:  # Gap to here too large, end line.
                break
            px += dx
            py += dy
    return n_line_pixels


def find_line2(x, y, mask, ls, lc, line_ends, line_pixels, line_gap,
               vprint):

    height = mask.shape[0]
    width = mask.shape[1]

    # From the random point (x, y), walk in opposite directions and
    # find line beginning and end (step 5 above).
    # Line equation is r = cos theta x + sin theta y.  Rearranging:
    # y = r / sin theta  - cos theta x / sin theta, and slope
    # is -cos theta / sin theta .
    slope = -lc / ls if ls else 99  # Marker value.
    sl_lt_1 = abs(slope) < 1
    if sl_lt_1:  # One of sin theta or cos theta must be non-zero.
        slope = ls / -lc
    # pass 1: walk the line, merging lines less than specified gap
    # length (step 5 continued).
    line_pixels[0, 0] = x
    line_pixels[0, 0] = y
    n_line_pixels = 1
    for k in range(2):
        gap = 0
        px = x
        py = y
        delta = -1 if k else 1
        offset = delta
        while True:
            if sl_lt_1:
                px = x + offset
                py = y + round(offset * slope)
            else:
                py = y + offset
                px = x + round(offset * slope)
            # check when line exits image boundary
            if px < 0 or px >= width or py < 0 or py >= height:
                break
            gap += 1
            if mask[py, px]:  # Hit remaining pixel, continue line.
                gap = 0
                line_ends[k, 0] = px
                line_ends[k, 1] = py
                # Record presence of in-mask pixel on line.
                line_pixels[n_line_pixels, 0] = px
                line_pixels[n_line_pixels, 1] = py
                n_line_pixels += 1
            elif gap > line_gap:  # Gap to here too large, end line.
                break
            offset += delta
    return n_line_pixels
