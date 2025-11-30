# First Feature: edge detection and texture analysis
# Detects shadows in images and compares textures inside and outside shadows

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# FEATURE EXTRACTION HELPERS
# ----------------------------------------------------------
def mean_std(arr):
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))

def percentile(arr, p):
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))

def lbp_entropy_np(lbp, region=None):
    vals = lbp[region>0].ravel() if region is not None else lbp.ravel()
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=256, range=(0,256), density=True)
    hist = hist + 1e-12
    return float(-np.sum(hist * np.log(hist)))

def component_stats_min(mask):
    # return only num_shadow_components, comp_area_p90, comp_peri2_over_area_p50
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    areas, peri2_over_area = [], []

    for i in range(1, num):
        # 0 is the background
        a = stats[i, cv.CC_STAT_AREA]
        areas.append(a)
        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv.findContours(comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts and a > 0:
            peri = cv.arcLength(max(cnts, key=cv.contourArea), True)
            peri2_over_area.append((peri * peri) / a)

    out = {"num_shadow_components": int(num - 1)}
    out["comp_area_p90"] = percentile(areas, 90)
    out["comp_peri2_over_area_p50"] = percentile(peri2_over_area, 50)
    return out

def angular_variance_min(gx, gy, boundary_mask, eps=1e-6):
    # return only: normal_mrl (alignment) and normal_angle_std (irregularity)
    by, bx = np.where(boundary_mask > 0)
    if by.size == 0:
        return {"normal_angle_std": 0.0, "normal_mrl": 0.0}
    
    gxs = gx[by, bx].astype(np.float32)
    gys = gy[by, bx].astype(np.float32)
    mag = np.sqrt(gxs * gxs + gys * gys) + eps
    nx, ny = gxs / mag, gys / mag
    ang = np.arctan2(ny, nx)
    C = float(np.mean(np.cos(ang)))
    S = float(np.mean(np.sin(ang)))
    R = float(np.sqrt(C * C + S * S))     # mean resultant length is [0..1]
    ang_std = float(np.sqrt(-2 * np.log(R + eps)))
    return {"normal_angle_std": ang_std, "normal_mrl": R}

def boundary_from_mask(mask):
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    return cv.morphologyEx(mask, cv.MORPH_GRADIENT, se)

# ----------------------------------------------------------
# LOCAL BINARY PATTERN (LBP) for texture analysis
# ----------------------------------------------------------
'''
look at each pixel and compare it to its 8 neighbors (top, bottom, left, right, diagonals);
if its neighbor is brighter, it is marked as 1, if darker then 0;
a binary pattern is created to describe the texture around a pixel
'''
def get_pixel(img, center, x, y):
    # check if neighbor pixels are brighter than the center pixel
    new_value = 0

    try:
        # if local neighbor pixel >= center pixel 
        if img[x][y] >= center:
            new_value = 1
    except:
        # return 0 if the pixel is at the edge of the image
        pass

    return new_value

def lbp_calculated_pixel(img, x, y): 
    # calculating LBP value for a single pixel by comparing it to 8 neighbors
    center = img[x][y]

    # create array of pixels based on the 8 neighbors
    val_arr = []
    val_arr.append(get_pixel(img, center, x-1, y-1)) # top left
    val_arr.append(get_pixel(img, center, x-1, y))   # top
    val_arr.append(get_pixel(img, center, x-1, y+1))  # top right
    val_arr.append(get_pixel(img, center, x, y+1))    # right
    val_arr.append(get_pixel(img, center, x+1, y+1))   # bottom right
    val_arr.append(get_pixel(img, center, x+1, y))    # bottom
    val_arr.append(get_pixel(img, center, x+1, y-1))    # bottom left
    val_arr.append(get_pixel(img, center, x, y-1))    # left
    
    # convert binary pattern values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]    
    val = 0
    for i in range(len(val_arr)):
        val += val_arr[i] * power_val[i]
    return val

def lbp_map(gray):
    # creating an LBP map for the image after turning it to grayscale
    height,width = gray.shape
    out = np.zeros((height,width), np.uint8)

    for i in range(height):
        for j in range(width):
            out[i,j] = lbp_calculated_pixel(gray, i, j)
    return out

def lbp_hist(patch):
    # 8-neighbor lbp -> values 0 to 255
    # histogram for local texture pattern distribution
    hist, _ = np.histogram(patch.ravel(), bins=256, range=(0,256), density=True)
    return hist

# ----------------------------------------------------------
# SHADOW MASK HELPERS
# ----------------------------------------------------------
def box_mean(arr, k):
    # calculating the average of neighboring pixels with box filter
    # good for reducing noise since it smooths
    return cv.boxFilter(arr, ddepth =- 1, ksize = (k, k), normalize = True)

def remove_small(mask, min_area = 800):
    # remove small noisy regions from the mask
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity = 8)
    keep = np.zeros_like(mask)
    
    # background is labeled 0
    # keep only the large regions
    for i in range(1, num):
        # use the total area (# of pixels) of the component
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255

    return keep

# RGB to HSI conversion
def bgr_to_hsi(img_bgr):
    img_bgr = img_bgr.astype(np.float32) / 255.0
    B, G, R = cv.split(img_bgr)
    
    # intensity
    I = (R + G + B) / 3.0
    
    # saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 1e-6, 1 - (min_rgb / (I + 1e-6)), 0)
    
    # hue (not needed for shadow detection)
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H * 180 / np.pi  # convert to degrees
    
    return H, S, I

def srgb_to_linear(x8):
    # convert the sRGB color space to linear space for better shadow detection 
    # sRGB is for visual/display, linear for math operations
    x = x8.astype(np.float32) / 255.0       # x in [0,1], sRGB -> linear
    # Normalize only if we're in 0..255 domain
    if x.max() > 1.0:
        x = x / 255.0
    a = 0.055
    return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def normalized_rgb(img_bgr):
    # normalize the RGB channels to help identify shadows through color consistency
    b, g, r = cv.split(img_bgr.astype(np.float32))

    # calculate sum of pixel values
    # total light intensity (similar to luminance) at each pixel location
    sum = r + g + b + 1e-6
    return r/sum, g/sum     # chroma ratios red and green since they are more stable than blue channel

def bgr_to_hsi_linear(img_bgr):
    # convert BGR to HSI color space (hue, saturation, intensity)
    # this is important to adjust brightness without affecting shadow's hue or saturation
    B, G, R = cv.split(img_bgr.astype(np.float32))

    # convert to linear space for calculations
    Rl = srgb_to_linear(R)
    Gl = srgb_to_linear(G)
    Bl = srgb_to_linear(B)

    # calculate the intensity (the average of rgb)
    I = (Rl + Gl + Bl) / 3.0

    # calculate the saturation (how colorful vs how gray)
    min_rgb = np.minimum(np.minimum(Rl, Gl), Bl)
    S = np.where(I > 1e-6, 1.0 - (min_rgb / (I + 1e-6)), 0.0)

    # hue not used since hue is not used in shadow detection 
    H = np.zeros_like(I, dtype=np.float32)
    return H, S, I

# ----------------------------------------------------------
# PREPROCESSING 
# ----------------------------------------------------------
def preprocess_for_shadow(
    img_bgr,
    gauss_ksize=5,
    gauss_sigma=1.0,
    log_ksize=3,
    log_sigma=0.0,
    log_alpha=0.35      
):
    '''
    applying preprocessing techniques of Gaussian blur and LoG to reduce noise and emphasis edges
    '''
    # gaussian blur on the color image
    if gauss_ksize < 3:
        gauss_ksize = 3
    if gauss_ksize % 2 == 0:
        gauss_ksize += 1
    img_blur = cv.GaussianBlur(img_bgr, (gauss_ksize, gauss_ksize), gauss_sigma)

    # LoG applied to grayscale
    gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
    if log_sigma > 0.0:
        if log_ksize < 3:
            log_ksize = 3
        if log_ksize % 2 == 0:
            log_ksize += 1
        gray_for_log = cv.GaussianBlur(gray, (log_ksize, log_ksize), log_sigma)
    else:
        gray_for_log = gray

    # laplacian on the (pre)blurred gray
    lap = cv.Laplacian(gray_for_log, cv.CV_32F, ksize=log_ksize if log_ksize >= 1 else 3)

    # normalize lap to a comparable scale and add to luminance
    # (log_alpha controls contribution)
    gray32 = gray.astype(np.float32)
    # scale Laplacian roughly into 8-bit
    lap_norm = lap / (np.max(np.abs(lap)) + 1e-6) * 127.0
    gray_enh = gray32 + log_alpha * lap_norm

    # clip back to [0,255] and cast
    gray_enh = np.clip(gray_enh, 0, 255).astype(np.uint8)

    # map enhanced gray back into BGR while keeping chroma roughly constant
    # compute ratio between enhanced and original luminance then apply to each channel
    denom = gray.astype(np.float32) + 1.0
    ratio = (gray_enh.astype(np.float32) + 1.0) / denom
    ratio = np.clip(ratio, 0.25, 4.0)  # avoid wild gains

    img_pre = img_blur.astype(np.float32)
    img_pre[..., 0] *= ratio
    img_pre[..., 1] *= ratio
    img_pre[..., 2] *= ratio

    img_pre = np.clip(img_pre, 0, 255).astype(np.uint8)
    return img_pre    

# ----------------------------------------------------------
# SHADOW MASK
# ----------------------------------------------------------
# mask to further distinguish shadows
def make_shadow_mask(
    img_bgr,
    beta=0.6,
    win_scales=(21, 41, 81),
    k_dark = (0.92, 0.95, 0.98),
    dr=0.06, dg=0.06,
    morph_open=5,
    morph_close=10,
    min_area=400,
):

    # ----- STEP 0: Light smoothing -----
    img_bgr = cv.GaussianBlur(img_bgr, (5, 5), 0)

    # ----- STEP 1: Convert to HSI Linear -----
    _, S, I = bgr_to_hsi_linear(img_bgr)

    # ----- STEP 2: Scene brightness statistics -----
    mean_I = np.mean(I)
    std_I  = np.std(I)
    avg_S  = np.mean(S)

    # ----- STEP 3: Dynamic k_dark (darker-than-local thresholds) -----
    contrast_factor = np.clip(std_I * 3.0, 0.6, 1.4)
    dark_base = 0.9 - (mean_I - 0.5) * 0.2
    dark_base = np.clip(dark_base, 0.82, 0.96)

    k_dark = (
        dark_base - 0.04 * contrast_factor,
        dark_base - 0.02 * contrast_factor,
        dark_base,
    )

    # ----- STEP 4: Local darkness detection -----
    meanI = [box_mean(I, w) for w in win_scales]
    darks = [(I < (kk * m)) for m, kk in zip(meanI, k_dark)]
    roi_dark = darks[0] | darks[1] | darks[2]

    if not np.any(roi_dark):
        return np.zeros(I.shape, np.uint8), (I * 255).astype(np.float32)

    # ----- STEP 5: Convert to luminance for intensity boost -----
    B, G, R = cv.split(img_bgr.astype(np.float32) / 255.0)
    Rl = srgb_to_linear(R * 255) / 255.0
    Gl = srgb_to_linear(G * 255) / 255.0
    Bl = srgb_to_linear(B * 255) / 255.0
    Y_lin = 0.114 * Bl + 0.587 * Gl + 0.299 * Rl

    # dynamic beta
    beta = np.clip(0.8 - (mean_I - 0.5) * 0.5, 0.5, 1.0)

    # intensity boost
    Iprime_raw = I + beta * Y_lin
    scale = float(np.percentile(Iprime_raw[roi_dark], 95))
    Iprime = np.clip(Iprime_raw / max(scale, 1e-6), 0.0, 1.0)

    # ----- STEP 6: Chroma consistency -----
    nr, ng = normalized_rgb(img_bgr)
    mnr = box_mean(nr, 41)
    mng = box_mean(ng, 41)

    # adaptive chroma tolerance
    dr = dg = np.clip(0.05 + (avg_S - 0.3) * 0.1, 0.03, 0.08)

    chroma_ok = (np.abs(nr - mnr) < dr) & (np.abs(ng - mng) < dg)

    # ----- STEP 7: Compute shadow likelihood (continuous) -----
    darkness_level = np.mean(I[roi_dark])

    sat_percentile = 30 + 40 * (1 - darkness_level)  # 30–70
    int_percentile = 40 + 30 * (1 - darkness_level)  # 40–70

    S_thr = float(min(0.4, np.percentile(S[roi_dark], sat_percentile) + 0.03))
    Ip_thr = float(np.percentile(Iprime[roi_dark], int_percentile))

    # likelihood score ∈ [0,1]
    shadow_likelihood = (
        (1 - np.clip(S / (S_thr + 1e-6), 0, 1)) *
        (1 - np.clip(Iprime / (Ip_thr + 1e-6), 0, 1))
    )

    # dynamic threshold selection by distribution
    shadow_cut = np.percentile(shadow_likelihood, 75)
    mask = (shadow_likelihood > shadow_cut).astype(np.uint8) * 255

    # ----- STEP 8: Dominant shadow direction -----
    gx = cv.Sobel(Iprime, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(Iprime, cv.CV_32F, 0, 1, ksize=3)
    ang = np.arctan2(gy, gx)

    # Use high likelihood pixels to estimate direction
    dir_mask = shadow_likelihood > shadow_cut
    if np.any(dir_mask):
        hist, bins = np.histogram(ang[dir_mask], bins=36, range=(-np.pi, np.pi))
        dominant_angle = bins[np.argmax(hist)]
    else:
        dominant_angle = 0.0

    # angular filtering (remove random-direction blobs)
    angle_diff = np.abs(np.angle(np.exp(1j * (ang - dominant_angle))))
    mask[(angle_diff > np.deg2rad(35))] = 0

    # ----- STEP 9: Conditional vegetation suppression -----
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    H, S_hsv, V_hsv = cv.split(hsv)

    green_candidates = ((H > 25) & (H < 90) & (S_hsv > 50))

    # greenery dark enough?
    local_mean = cv.boxFilter(V_hsv.astype(np.float32), -1, (15, 15))
    green_dark = V_hsv.astype(np.float32) < (0.92 * local_mean)

    # greenery aligned with shadow direction?
    green_dir_ok = angle_diff < np.deg2rad(35)

    # INVALID GREEN = leafy blobs NOT aligned with sun & not dark enough
    invalid_green = green_candidates & ~(green_dark & green_dir_ok)
    mask[invalid_green] = 0

    # ----- STEP 10: Morphological cleanup -----
    h, w = img_bgr.shape[:2]
    scale_factor = np.sqrt((h * w) / (1080 * 1920))

    morph_open = max(3, int(morph_open * scale_factor))
    morph_close = max(7, int(morph_close * scale_factor))
    min_area = int(min_area * scale_factor**2)

    if morph_open > 1:
        k1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k1)

    if morph_close > 1:
        k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k2)

    mask = remove_small(mask, min_area=min_area)
    mask = cv.GaussianBlur(mask, (5, 5), 0)

    return mask, (I * 255).astype(np.float32)
 

# ----------------------------------------------------------
# TEXTURE COMPARISON AND HELPERS FOR TAMPER SCORE
# compare texture between shadow patches and nearby lit patches
# ----------------------------------------------------------
def chi2(a, b, eps=1e-9):
    # chi-squared distance between two histograms (measuring texture simularity)
    d = (a - b)
    s = (a + b) + eps
    return 0.5 * np.sum((d * d) / s)

def pairs_one_per_component(
    mask_u8, L8, lbp, gx, gy,
    min_area=300,            # keep consistent with your mask cleanup
    patch_size=25, 
    in_offset=6, 
    out_offset=9, 
    max_step=30
):
    '''
    for each connected shadow component in mask_u8, pick one pair of patches to compare:
      - inside patch: near boundary but inside (stable, not on edge)
      - outside patch: just outside the boundary in the lit area
    return:
      chi2_list (distances): list[float]
      patch_pairs: [((x_in,y_in),(x_out,y_out), size)]
    '''
    h, w = L8.shape
    s = patch_size // 2
    chi2_list, patch_pairs = [], []

    # find all separate shadow components/regions
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask_u8, connectivity=8)
    k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    def in_bounds(y, x):
        # check if patch fits in image
        return (y - s) >= 0 and (x - s) >= 0 and (y + s) < h and (x + s) < w

    def crop(img, y, x):
        # extract patch from image
        return img[y - s:y + s + 1, x - s:x + s + 1]

    # go through each shadow region
    for i in range(1, num):  # 0 is background
        area = stats[i, cv.CC_STAT_AREA]
        # skip small regions
        if area < min_area:
            continue

        # grab shadow component    
        comp = (labels == i).astype(np.uint8) * 255
        boundary = cv.morphologyEx(comp, cv.MORPH_GRADIENT, k3)
        
        if boundary.sum() == 0:
            continue

        # find a good point inside the shadow that is far from edges
        dist = cv.distanceTransform((comp > 0).astype(np.uint8), cv.DIST_L2, 5)
        yi_in, xi_in = np.unravel_index(np.argmax(dist), dist.shape)
        yi_in = int(np.clip(yi_in, s, h - s - 1))
        xi_in = int(np.clip(xi_in, s, w - s - 1))
        
        if not in_bounds(yi_in, xi_in):
            continue

        # find the closest boundary point
        by, bx = np.where(boundary > 0)
        # calculate the squared euclidean distance between a boundary point with coords bx, by and each point ->
        # in a set of points with coords xi_in, yi_in;
        # the index of the point with closest distance to bx, by is assigned to j
        j = int(np.argmin((by - yi_in)**2 + (bx - xi_in)**2))
        yb, xb = int(by[j]), int(bx[j])

        # find out which direction is outward (going toward lit area)
        g = np.array([gx[yb, xb], gy[yb, xb]], dtype=np.float32)
        nrm = float(np.linalg.norm(g))

        if nrm > 1e-3:
            # gradient direction
            nx, ny = g[0] / nrm, g[1] / nrm
        else:
            # use direction from center to boundary
            M = cv.moments(comp, binaryImage=True)
            # total area in shadow; if area is 0 (empty shadow), use bound point as center
            if abs(M["m00"]) < 1e-6:
                cx, cy = xb, yb
            else:
                # calculate actual center
                # average x = sum of all x values / num of pixels
                cx = M["m10"] / M["m00"] 
                cy = M["m01"] / M["m00"]
            v = np.array([xb - cx, yb - cy], dtype=np.float32)
            vn = np.linalg.norm(v)
            nx, ny = (v / vn) if vn > 1e-6 else (1.0, 0.0)

        # get the inside shadow patch
        yi_samp = int(np.clip(yb - in_offset * ny, s, h - s - 1))
        xi_samp = int(np.clip(xb - in_offset * nx, s, w - s - 1))
        
        if not in_bounds(yi_samp, xi_samp):
            continue

        # get the outside shadow patch
        step = out_offset
        yo = int(np.clip(yb + step * ny, s, h - s - 1))
        xo = int(np.clip(xb + step * nx, s, w - s - 1))
        
        # keep moving outward until outside the shadow
        while step < max_step and comp[yo, xo] > 0:
            step += 2
            yo = int(np.clip(yb + step * ny, s, h - s - 1))
            xo = int(np.clip(xb + step * nx, s, w - s - 1))
        
        if not in_bounds(yo, xo):
            continue

        # extract patches
        pinL = crop(L8,  yi_samp, xi_samp)
        poutL = crop(L8,  yo, xo)
        pinLBP = crop(lbp, yi_samp, xi_samp)
        poutLBP = crop(lbp, yo, xo)

        # make sure the inside patch is darker than outside
        m_in = float(np.mean(pinL))
        m_out = float(np.mean(poutL))

        if m_out <= m_in:
            # swap to keep outside brighter
            pinL, poutL = poutL, pinL
            pinLBP, poutLBP = poutLBP, pinLBP
            (xi_samp, yi_samp), (xo, yo) = (xo, yo), (xi_samp, yi_samp)
            m_in, m_out = m_out, m_in

        # compare the textures with chi-squared distasnce
        h_in,  _ = np.histogram(pinLBP.ravel(),  bins=256, range=(0, 256), density=True)
        h_out, _ = np.histogram(poutLBP.ravel(), bins=256, range=(0, 256), density=True)
        
        # calculate the chi-squared distance (lower = more similar)
        '''
        ideal range for chi-squared distance:
        - 0-0.2: very similar textures (likely real shadows)
        - 0.2-0.5: moderately similar (could be real shadows with some variation)
        - 0.5-1.0: different textures (suspicious, might be fake)
        - >1.0: very different textures (likely fake; different materials)
        for our detection:
        - <0.3: likely a shadow
        - 0.3-0.8: some texture difference and may need more investigation
        ->0.8: likely a fake shadow
        '''
        d = 0.5 * np.sum(((h_in - h_out) ** 2) / (h_in + h_out + 1e-9))

        chi2_list.append(float(d))
        x0_in, y0_in = xi_samp - s, yi_samp - s
        x0_out, y0_out = xo - s, yo - s
        patch_pairs.append(((x0_in, y0_in), (x0_out, y0_out), patch_size))

    return chi2_list, patch_pairs

# ----------------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------------
def extract_features(
    img_bgr,
    mask, 
    lbp, 
    L8,
    gx, 
    gy, 
    chi2_list        
) -> dict:
    '''
    to build feature vector:
    - mask (uint8 0/255),
    - lbp (uint8),
    - L8 (uint8 luminance),
    - gx, gy (float32 for Sobel gradients),
    - chi2_list (contains chi-squared distances comparing LBP patches inside vs outside)
    expected inputs: intermediates from texture.py and per-boundary sample arrays
    '''
    ml_features = {}

    # boundary and masks
    mask = mask.astype(np.uint8)
    boundary = boundary_from_mask(mask)

    in_m = mask > 0
    out_m = ~in_m

    # coverage/edges
    '''
    mask_frac = how much of the image is covered by a detected shadow
    boundary_frac = how long or detailed the shadow edge is compared to the whole image
    '''
    ml_features["mask_frac"] = float(in_m.mean())
    ml_features["boundary_frac"] = float((boundary > 0).mean())

    # luminance contrast (means/stds only)
    '''
    L_in_mean/L_out_mean = average brightness inside and outside the shadow
    (shadows should be darker)
    L_in_std/L_out_std = how much brightness varies (contrast inside and outside)
    (real shadows keep surface texture, fake ones look smooth)
    contrast_shadow_vs_non = how strong the brightness drop is from light to shadow
    (ratio of how dark the shadow region is compared to the lit area)
    '''
    if in_m.any():
        mu_in, sd_in = mean_std(L8[in_m])
    else:
        mu_in, sd_in = 0.0, 0.0
    if out_m.any():
        mu_out, sd_out = mean_std(L8[out_m])
    else:
        mu_out, sd_out = 0.0, 0.0 

    ml_features["L_in_mean"] = mu_in
    ml_features["L_in_std"]  = sd_in
    ml_features["L_out_mean"] = mu_out
    ml_features["L_out_std"]  = sd_out
    ml_features["contrast_shadow_vs_non"] = (mu_out - mu_in) / (mu_out + 1e-6)

    # texture entropy (LBP)
    '''
    entropy = measure of randomness in the texture pattern (higher = more detailed texture)
    inside vs outside entropy comparison tells whether the shadowed region kept the same --
    texture randomness as the lit area
    '''
    ml_features["lbp_entropy_in"]  = lbp_entropy_np(lbp, region=mask)
    ml_features["lbp_entropy_out"] = lbp_entropy_np(lbp, region=(255 - mask))

    # shadow component shape (minimal)
    '''
    look at shape of the shadow areas
    - how many separate shadow blobs exist
    - how large they are
    - how irregular the boundaries are
    - ^ helps describe whether the detected shadow is one smooth region or lots of tiny parts (noisy)
    '''
    ml_features.update(component_stats_min(mask))

    # boundary geometry consistency
    '''
    use gradient directions gx and gy to see how consistent the shadow edge direction is
    - if all edge directions are aligned, clear and natural boundary
    - if edge directions point everywhere, messy or fake
    '''
    ml_features.update(angular_variance_min(gx, gy, boundary))

    '''
    how similar or different the textures are between the shadow side and a nearby lit side
    - chi2_count = how many patch comparisons made
    - chi2_mean = average of how different textures are across the shadow edge
    - chi2_std = how much of the differences vary from one place to another
    - chi2_p25/p50/p75  = low, middle, and high ranges of texture differences
    - share_low_chi2 = fraction of edges where textures look very similar (likely real)
    - share_high_chi2 = fraction of edges where textures look very different (possible fake)
    '''
    d_arr = np.asarray(chi2_list, dtype=np.float32)
    ml_features["chi2_count"] = int(d_arr.size)
    ml_features["chi2_mean"] = float(d_arr.mean()) if d_arr.size else 0.0
    ml_features["chi2_std"] = float(d_arr.std()) if d_arr.size else 0.0
    ml_features["chi2_p25"] = percentile(d_arr, 25) if d_arr.size else 0.0
    ml_features["chi2_p50"] = percentile(d_arr, 50) if d_arr.size else 0.0
    ml_features["chi2_p75"] = percentile(d_arr, 75) if d_arr.size else 0.0

    # how often similarity is high (small chi-squared value) or low (large chi-squared value)
    '''
    low std = region looks smooth (possibly fake shadow)
    high std = region has texture detail (typical of real shadow)
    '''
    ml_features["share_low_chi2"]  = float(np.mean(d_arr <= (ml_features["chi2_p25"] if d_arr.size else 0.0))) if d_arr.size else 0.0
    ml_features["share_high_chi2"] = float(np.mean(d_arr >= (ml_features["chi2_p75"] if d_arr.size else 0.0))) if d_arr.size else 0.0

    return ml_features

# ----------------------------------------------------------
# TAMPER SCORE CALCULATION & FEATURE EXTRACTION
# ----------------------------------------------------------
def calculate_texture_tamper_score(chi2_list):
    '''
    tamper score between 0 to 1 based on texture comparisons of each pair of patches for each shadow component:
    0.0 = likely real shadows (textures match)
    0.1 = likely fake shadows (textures are very different)

    considering:
    - how different the textures are (chi-squared distances for similarity)
    - how many suspicious shadow regions there are
    - consistency across all shadows
    '''
    if len(chi2_list) == 0:
        return 0.0  # no shadows are found so assume real
    
    chi2_arr = np.array(chi2_list)

    # thresholds
    LOW_THRESHOLD = 0.10   # similar texture
    HIGH_THRESHOLD = 0.30   # very different texture

    # average texture difference for first score component
    mean_chi2 = float(np.mean(chi2_arr))
    if mean_chi2 <= LOW_THRESHOLD:
        score_mean = 0.0
    elif mean_chi2 >= HIGH_THRESHOLD:
        score_mean = 1.0
    else:
        # linear interpolation if mean is between thresholds
        score_mean = (mean_chi2 - LOW_THRESHOLD) / (HIGH_THRESHOLD - LOW_THRESHOLD)

    # max difference (worst case) for second score component
    # flag if even one shadow is very suspicious
    max_chi2 = float(np.max(chi2_arr))
    if max_chi2 <= LOW_THRESHOLD:
        score_max = 0.0
    elif max_chi2 >= HIGH_THRESHOLD:
        score_max = 1.0
    else:
        score_max = (max_chi2 - LOW_THRESHOLD) / (HIGH_THRESHOLD - LOW_THRESHOLD)

    # consistency of values for third score component
    # real shadows should have similar values
    # fake shadows might have mixed results (some match, some don't)
    std_chi2 = float(np.std(chi2_arr))
    if std_chi2 > 0.1:
        score_consistency = 0.3  # penalty for inconsistency
    else:
        score_consistency = 0.0
    
    # percentage of suspicious shadows for fourth score component
    # count how many shadows exceed the high threshold
    num_suspicious = int(np.sum(chi2_arr > HIGH_THRESHOLD))
    pct_suspicious = num_suspicious / len(chi2_arr)
    score_percentage = pct_suspicious
    
    # combine all components with weights
    # mean is most important, max catches extreme cases
    tamper_score = (
        0.40 * score_mean +        # average behavior
        0.30 * score_max +         # worst case
        0.15 * score_consistency + # how consistent
        0.15 * score_percentage    # how many are bad
    )
    
    # clamp score to [0, 1]
    tamper_score = max(0.0, min(1.0, tamper_score))
    
    return tamper_score           

# ----------------------------------------------------------
# VISUALIZATION HELPERS
# ----------------------------------------------------------
def visualize_texture_analysis(img, mask, L8, lbp, chi2_list, patch_pairs, max_pairs_vis=200):
    '''
    visualize the texture analysis results with maps
    '''
    # outline of the mask (shadow boundaries)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask_outline = cv.morphologyEx(mask, cv.MORPH_GRADIENT, k)

    # make LBP texture map 
    lbp = lbp_map(L8)
    lbp_vis = cv.normalize(lbp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    # outline the shadow areas (using mask boundaries)
    lbp_color = cv.applyColorMap(lbp_vis, cv.COLORMAP_BONE)
    lbp_color[mask_outline > 0] = (255, 0, 0)

    # show shadow mask
    mask_overlay = img.copy()
    mask_overlay[mask == 255] = (0, 0, 255)  # red mask overlay
    mask_overlay = cv.addWeighted(img, 0.7, mask_overlay, 0.3, 0)

    cv.namedWindow("Shadow Mask", cv.WINDOW_NORMAL)
    cv.resizeWindow("Shadow Mask", 800, 600)
    cv.imshow("Shadow Mask", mask)

    cv.namedWindow("Overlay", cv.WINDOW_NORMAL)
    cv.resizeWindow("Overlay", 800, 600)
    cv.imshow("Overlay", mask_overlay)

    cv.waitKey(1)

    # show lbp map with matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(lbp_vis, cmap='gray')
    plt.contour(mask_outline > 0, colors='red', linewidths=0.5)
    plt.title("LBP with bounds in red")
    plt.show(block=False)
    plt.pause(0.001)
    plt.show()

    # patch pair rectangles (inside=red, outside=green)
    overlay_pairs = lbp_color.copy()
    vis_n = min(len(patch_pairs), max_pairs_vis)

    for i in range(vis_n):
        (x_in, y_in), (x_out, y_out), s = patch_pairs[i]
        # red box = inside shadow
        cv.rectangle(overlay_pairs, (x_in, y_in), (x_in + s, y_in + s), (0, 0, 255), 2)
        # green box = outside shadow (lit area)
        cv.rectangle(overlay_pairs, (x_out, y_out), (x_out + s, y_out + s), (0, 255, 0), 2)
        # label pairs
        cv.putText(overlay_pairs, str(i), (x_in, max(0, y_in - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv.LINE_AA)
        cv.putText(overlay_pairs, str(i), (x_out, max(0, y_out - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv.LINE_AA)                

    cv.namedWindow("Comparison Patches (red=inside shadow, green=outside shadow)", cv.WINDOW_NORMAL)
    cv.resizeWindow("Comparison Patches (red=inside shadow, green=outside shadow)", 1000, 750)
    cv.imshow("Comparison Patches (red=inside shadow, green=outside shadow)", overlay_pairs)

    # chi-squared histogram (similarity scores)
    if len(chi2_list) > 0:
        plt.figure(figsize=(8,5))
        plt.hist(chi2_list, bins=40)
        plt.title("Texture Similarity Scores - LBP chi-squared distances")
        plt.xlabel("Chi-squared distance (lower = more similar)")
        plt.ylabel("Number of shadow regions")
        plt.tight_layout()
        plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()

# ----------------------------------------------------------
# CHOOSING THE IMAGE - MAIN ANALYSIS
# ----------------------------------------------------------
def analyze_texture(image_input, visualize=True, compute_tamper_score=True, max_pairs_vis=200):
    '''
    analyze texture across shadow boundaries using LBP and return chi-square similarities
    - highlight the pairs of patches (one pair per shadow) that are compared
    - return raw chi-square distances for ML feature engineering
    '''
    if isinstance(image_input, str):
        img = cv.imread(image_input)
    else:
        img = image_input

    assert img is not None, f"Cannot read image: {image_input}"

    # detect shadows with shadow mask
    mask,L = make_shadow_mask(
        img, 
        beta = 0.8,  # trying 0.6-0.8 (street level) or 0.8-1.0 (aerial)
        win_scales = (21, 41, 81),
        k_dark = (0.92, 0.95, 0.98),
        dr=0.06, dg=0.06,
        morph_open = 3, 
        morph_close = 7, 
        min_area = 300
    )
    assert mask.shape[:2] == img.shape[:2], "Mask must align with original image"
    # convert intensity to 8-bit
    # create texture inpts from the original image, not the preprocessed image
    L8_mask = np.uint8(np.clip(L, 0, 255))
    
    _, _, I0 = bgr_to_hsi_linear(img)
    L8_text = np.uint8(np.clip(I0*255, 0, 255))

    # calculate gradients to find outward direction of shadow
    gx = cv.Sobel(L8_text, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(L8_text, cv.CV_32F, 0, 1, ksize=3)

    # make LBP texture map 
    lbp = lbp_map(L8_text)

    # EXACTLY one pair per component
    chi2_list, patch_pairs = pairs_one_per_component(
        mask_u8=mask, 
        L8=L8_text, 
        lbp=lbp, 
        gx=gx, 
        gy=gy,
        min_area=300, 
        patch_size=25, 
        in_offset=6, 
        out_offset=9
    )

    print("\nTexture similarity scores (chi-squared distance):")
    print("Lower values = more similar texture = likely real shadow")
    print("Higher values = different texture = possible fake shadow")
    for i, d in enumerate(chi2_list, start=1):
        print(f"  Pair {i:02d}: {d:.4f}")

    # extract features for ML (independent of rule-based scoring)
    features = extract_features(
        img,
        mask=mask,
        lbp=lbp,
        L8=L8_text, 
        gx=gx,
        gy=gy,
        chi2_list=chi2_list,
    )

    # calculate tamper score (rule-based only)
    tamper_score = None
    if compute_tamper_score:
        tamper_score = calculate_texture_tamper_score(chi2_list)
        print(f"\n{'='*60}")
        print(f"TEXTURE TAMPER SCORE: {tamper_score:.3f}")
        print(f"{'='*60}")

    if visualize:
        # outline of the mask (shadow boundaries)
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        mask_outline = cv.morphologyEx(mask, cv.MORPH_GRADIENT, k)

        # lbp map
        lbp_vis = cv.normalize(lbp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        # outline the shadow areas (using mask boundaries)
        lbp_color = cv.applyColorMap(lbp_vis, cv.COLORMAP_BONE)
        lbp_color[mask_outline > 0] = (255, 0, 0)

        # show shadow mask
        mask_overlay = img.copy()
        mask_overlay[mask == 255] = (0, 0, 255)  # red mask overlay
        mask_overlay = cv.addWeighted(img, 0.7, mask_overlay, 0.3, 0)

        cv.namedWindow("Shadow Mask", cv.WINDOW_NORMAL)
        cv.resizeWindow("Shadow Mask", 800, 600)
        cv.imshow("Shadow Mask", mask)

        cv.namedWindow("Overlay", cv.WINDOW_NORMAL)
        cv.resizeWindow("Overlay", 800, 600)
        cv.imshow("Overlay", mask_overlay)

        cv.waitKey(1)

        # show lbp map with matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(lbp_vis, cmap='gray')
        plt.contour(mask_outline > 0, colors='red', linewidths=0.5)
        plt.title("LBP with bounds in red")
        plt.show(block=False)
        plt.pause(0.001)
        plt.show()

        # patch pair rectangles (inside=red, outside=green)
        overlay_pairs = lbp_color.copy()
        vis_n = min(len(patch_pairs), max_pairs_vis)

        for i in range(vis_n):
            (x_in, y_in), (x_out, y_out), s = patch_pairs[i]
            # red box = inside shadow
            cv.rectangle(overlay_pairs, (x_in, y_in), (x_in + s, y_in + s), (0, 0, 255), 2)
            # green box = outside shadow (lit area)
            cv.rectangle(overlay_pairs, (x_out, y_out), (x_out + s, y_out + s), (0, 255, 0), 2)
            # label pairs
            cv.putText(overlay_pairs, str(i), (x_in, max(0, y_in - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv.LINE_AA)
            cv.putText(overlay_pairs, str(i), (x_out, max(0, y_out - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv.LINE_AA)                

        cv.namedWindow("Comparison Patches (red=inside shadow, green=outside shadow)", cv.WINDOW_NORMAL)
        cv.resizeWindow("Comparison Patches (red=inside shadow, green=outside shadow)", 1000, 750)
        cv.imshow("Comparison Patches (red=inside shadow, green=outside shadow)", overlay_pairs)

        # chi-squared histogram (similarity scores)
        if len(chi2_list) > 0:
            plt.figure(figsize=(8,5))
            plt.hist(chi2_list, bins=40)
            plt.title("Texture Similarity Scores - LBP chi-squared distances")
            plt.xlabel("Chi-squared distance (lower = more similar)")
            plt.ylabel("Number of shadow regions")
            plt.tight_layout()
            plt.show()

        cv.waitKey(0)
        cv.destroyAllWindows()

    # return results
    result = {
        "mask": mask, 
        "lbp": lbp,
        "L8": L8_text, 
        "chi2_distances": chi2_list,
        "patch_pairs": patch_pairs,
        "features": features,
    }

    # only include tamper_score if computed
    if tamper_score is not None:
        result["tamper_score"] = tamper_score

    return result


if __name__ == "__main__":
    result = analyze_texture("data/images/32.jpg", visualize=True)
    print("\nExtracted features:", result["features"])
