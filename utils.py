import os
import shutil
import numpy as np
import cv2


def visualize_png(img, indicator, visu_pngs_dir, flag=True):
    if flag:
        saved_path = os.path.join(visu_pngs_dir, indicator + ".png")
        if img.dtype == np.float32 or img.dtype == np.float64:
            saved_img = img * 255
            cv2.imwrite(saved_path, saved_img)
        elif img.dtype == np.uint8:
            img = img
            cv2.imwrite(saved_path, img)
        else:
            print("img dtype is not float32 or uint8, please check your code.")


def zero_kernel_centre(kernel, centre_radius):
    # Set the kernel center to 0.
    if centre_radius == 0:
        return kernel
    centre = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if (i - centre) ** 2 + (j - centre) ** 2 <= centre_radius ** 2:
                kernel[i, j] = 0
    return kernel
def circular_avg_kernel(radius):
    # Return a circular mean filter.
    diameter = int(2 * radius)
    if diameter % 2 == 0:
        diameter += 1
    centre = diameter // 2
    kernel = np.zeros([diameter, diameter], dtype=np.float32)
    for i in range(diameter):
        for j in range(diameter):
            if (i - centre) ** 2 + (j - centre) ** 2 <= radius ** 2:
                kernel[i, j] = 1
    return kernel / kernel.sum()
def get_gaussian_kernel_size(sigma):
    # Return an integer that represents the size of Gaussian kernel
    threshold = 1e-2
    radius = np.sqrt(-np.log(np.sqrt(2 * np.pi) * sigma * threshold) * (2 * sigma * sigma))
    radius = int(np.ceil(radius))
    kernel_size = radius * 2 + 1
    return kernel_size
def gaussian_gradient_kernel(sigma, theta, seta):
    # Return the Gaussian gradient with orientation theta and spatial aspect ratio seta.
    k_size = get_gaussian_kernel_size(sigma)
    kernel = np.zeros([k_size, k_size], dtype=np.float32)
    sqr_sigma = sigma ** 2
    width = k_size // 2
    for i in range(k_size):
        for j in range(k_size):
            y1 = i - width
            x1 = j - width
            x = x1 * np.cos(theta) + y1 * np.sin(theta)
            y = - x1 * np.sin(theta) + y1 * np.cos(theta)
            kernel[i, j] = - x * np.exp(-(x ** 2 + y ** 2 * seta ** 2) / (2 * sqr_sigma)) / (np.pi * sqr_sigma)

    # max_v = kernel.max()
    # kernel[kernel < 0.01 * max_v] = 0
    return kernel
def ellipse_gaussian_kernel(sigma_x, sigma_y, theta):
    # Return an ellipse gaussian kernel with orientation theta.
    max_sigma = max(sigma_x, sigma_y)
    kernel_size = get_gaussian_kernel_size(max_sigma)

    centre_x = kernel_size // 2
    centre_y = kernel_size // 2
    sqr_sigma_x = sigma_x ** 2
    sqr_sigma_y = sigma_y ** 2

    a = np.cos(theta) ** 2 / (2 * sqr_sigma_x) + np.sin(theta) ** 2 / (2 * sqr_sigma_y)
    b = - np.sin(2 * theta) / (4 * sqr_sigma_x) + np.sin(2 * theta) / (4 * sqr_sigma_y)
    c = np.sin(theta) ** 2 / (2 * sqr_sigma_x) + np.cos(theta) ** 2 / (2 * sqr_sigma_y)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - centre_x
            y = j - centre_y
            kernel[i, j] = np.exp(- (a * x ** 2 + 2 * b * x * y + c * y ** 2))

    # max_v = kernel.max()
    # kernel[kernel < 0.01 * max_v] = 0
    return kernel / kernel.sum()
def late_inhi_kernel(sigma_x, sigma_y, delta_degree, theta):
    """
        This function first call ellipse_gaussian_kernel() with orientation theta to get the orient_kernel.
        Then rotate the orient_kernel, substract the orient_kernel, and compute the maximal non-zero value at each location.
        At last, return the lateral inhibition kernel.
    """
    orient_kernel = ellipse_gaussian_kernel(sigma_x, sigma_y, theta=theta)

    n_degree = 180 // delta_degree
    orient_kernels = []
    for idx in range(3 * n_degree):
        k = ellipse_gaussian_kernel(sigma_x, sigma_y, theta=idx * delta_degree / 3 / 180 * np.pi)
        k = k - orient_kernel
        k[k < 0] = 0
        orient_kernels.append(k)

    kernel = np.max(np.array(orient_kernels), axis=0)
    return kernel / kernel.sum()
def texture_gradient_kernel(theta, radius):
    # Return the texture gradient kernel with orientation theta.
    kernel = circular_avg_kernel(radius)
    center = int(np.floor(radius))

    # 1. top half circle
    top_half_kernel = np.copy(kernel)
    top_half_kernel[center:, :] = 0
    M = cv2.getRotationMatrix2D((kernel.shape[0] // 2, kernel.shape[1] // 2), -theta, 1)
    top_half_kernel = cv2.warpAffine(top_half_kernel, M, (kernel.shape[1], kernel.shape[0]))

    # 2. bottom half circle
    bottom_half_kernel = np.copy(kernel)
    bottom_half_kernel[:center + 1, :] = 0
    M = cv2.getRotationMatrix2D((kernel.shape[0] // 2, kernel.shape[1] // 2), -theta, 1)
    bottom_half_kernel = cv2.warpAffine(bottom_half_kernel, M, (kernel.shape[1], kernel.shape[0]))

    result_kernel = top_half_kernel - bottom_half_kernel
    result_kernel = result_kernel - kernel / radius
    return result_kernel


def get_on_off(x, std_r, thresh):
    on_off = x / std_r
    on_off[on_off >= thresh] = 1
    on_off[on_off < thresh] = 0
    return on_off
def get_local_max(x, size):
    half_size = size // 2
    temp = np.zeros([x.shape[0] + half_size * 2, x.shape[1] + half_size * 2], dtype=np.float32)
    temp[half_size:x.shape[0] + half_size, half_size:x.shape[1] + half_size] = x
    result = np.zeros(x.shape, dtype=np.float32)
    circle_mask = circular_avg_kernel(size / 2)
    circle_mask /= circle_mask.max()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            local_area = temp[i:i + half_size + half_size + 1, j:j + half_size + half_size + 1] * circle_mask
            max_val = local_area.max()
            result[i, j] = max_val
    return result
def Relu_thresh(x, thresh):
    x[x < thresh] = 0
    return x


class EdgeSegment:
    def __init__(self):
        self.key_points = None
        self.val = 0
        self.pattern_idx = None
def find_key_point(A, B):
    t = A * B
    pos = np.argwhere(t == t.max())
    return pos[0], t.max()
def DDA(h1, w1, h2, w2):
    # A naive way of drawing line
    edge = []
    # If these tow points are same, add it and return
    if w1 == w2 and h1 == h2:
        edge.append([h1, w1])
        return edge
    # If these two points are neighbors in 8-neighbors
    if abs(h2 - h1) <= 1 and abs(w2 - w1) <= 1:
        edge.append([h1, w1])
        edge.append([h2, w2])
        return edge
    # Apply the DDA algorithm
    dw = w2 - w1
    dh = h2 - h1
    # Slope
    if abs(dw) > abs(dh):
        steps = abs(dw)
    else:
        steps = abs(dh)
    # One is equal to 1, and another less than 1
    delta_w = dw / steps
    delta_h = dh / steps
    # Round to the nearest, ensuring that the increments of 'w' and 'h' are less than or equal to 1,
    # to make the generated lines as evenly distributed as possible
    w = w1 + 0.5
    h = h1 + 0.5
    for i in range(0, int(steps + 1)):
        # Add points
        edge.append([int(h), int(w)])
        w = w + delta_w
        h = h + delta_h
    # Note that the DDA algorithm might reorder the points on the edge and requires rearrangement.
    if edge[0][0] != h1 or edge[0][1] != w1:
        edge.reverse()
    return edge


def discard_tiny_weight(kernel, max_val, thresh):
    if max_val <= 0:
        print("kernel is wrong")
        return kernel
    kernel = kernel / max_val
    kernel[kernel < thresh] = 0
    return kernel
def pad_kernel(kernel, size):
    result = np.zeros([size, size], dtype=np.float32)
    center = size // 2
    half_kernel_size = kernel.shape[0] // 2
    result[center-half_kernel_size:center+half_kernel_size+1, center-half_kernel_size:center+half_kernel_size+1] = kernel
    return result
def pad_tensor_2d(tensor, half_kernel):
    shape = tensor.shape
    temp = np.zeros([shape[0] + 2 * half_kernel, shape[1] + 2 * half_kernel], dtype=np.float32)
    temp[half_kernel:half_kernel+shape[0], half_kernel:half_kernel+shape[1]] = tensor
    return temp


def coloring_hue_s1_pixel(hue):
    # Compute the RGB cprresponding to the hue
    rgb = np.zeros([3], dtype=np.uint8)
    if hue <= 30 or hue > 300:
        if hue <= 30:
            dis = hue / 30  # yb_dis_factor

            r = 1
            b = 0
            g = dis / (dis + 1)
        else:
            dis = (360 - hue) / 60  # by_dis_factor

            r = 1
            g = 0
            b = dis
    elif hue > 30 and hue <= 90:
        if hue <= 60:
            dis = (60 - hue) / 30  # rg_dis_factor

            r = 1
            b = 0
            g = 1 / (dis + 1)
        else:
            dis = (hue - 60) / 30  # gr_dis_factor

            g = 1
            b = 0
            r = 1 / (dis + 1)
    elif hue > 90 and hue <= 180:
        if hue <= 120:
            dis = (120 - hue) / 30  # yb_dis_factor

            g = 1
            b = 0
            r = dis / (dis + 1)
        else:
            dis = (hue - 120) / 60  # by_dis_factor

            g = 1
            r = 0
            b = dis
    else:
        if hue <= 240:
            dis = (240 - hue) / 60  # gr_dis_factor

            b = 1
            r = 0
            g = dis
        else:
            dis = (hue - 240) / 60  # rg_dis_factor

            b = 1
            g = 0
            r = dis

    rgb[0] = np.uint8(b * 255)
    rgb[1] = np.uint8(g * 255)
    rgb[2] = np.uint8(r * 255)

    return rgb
def draw_continuous_edge(CS, input_img):
    # Prepare some colors
    hues = [0, 60, 120, 180, 300]
    num_hues = len(hues)
    color_list = [coloring_hue_s1_pixel(hue) for hue in hues]

    h, w = input_img.shape[0], input_img.shape[1]
    result = np.zeros([h, w], dtype=np.float32)
    result_coloring = np.zeros([h, w, 3], dtype=np.float32)

    # Color each edge segment
    for idx, cs in enumerate(CS):
        if len(cs.key_points) <= 20:
            continue
        color_idx = np.random.randint(num_hues)
        # The value of edge segment
        val = cs.val
        c = color_list[color_idx] * val / 255
        for point in cs.key_points:
            result_coloring[point[0], point[1], :] = c
            input_img[point[0], point[1], :] = c
            result[point[0], point[1]] = val
    return result, result_coloring, input_img


class LocalPattern:
    def __init__(self):
        self.k = None       # kernel,                       H X W X C
        self.k_l = None     # kernel_left_end-stopping,     H X W X C
        self.k_r = None     # kernel_right_end-stopping,    H X W X C
        self.k_f = None     # kernel_faci,                  H X W X C

        self.k_thresh = 0   # threshold for simple cells
        self.lr_thresh = 0  # threshold for end-stopping cells
        # Mask for searching
        self.l_link_mask = None
        self.r_link_mask = None
def isLeft_line(Ax, Ay, Bx, By, Px, Py):
    # Determine if P is on the left side of AB
    return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax) > 1e-4
def isRight_line(Ax, Ay, Bx, By, Px, Py):
    # Determine if P is on the right side of AB
    return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax) < -1e-4
def line_kernel(sigma_x, sigma_y, length, theta):
    kernel_size = length + 4

    kernel = np.zeros([kernel_size, kernel_size], dtype=np.float32)
    kernel_l_part = np.zeros([kernel_size, kernel_size], dtype=np.float32)
    kernel_r_part = np.zeros([kernel_size, kernel_size], dtype=np.float32)
    kernel_dis = np.zeros([kernel_size, kernel_size], dtype=np.float32)

    sqr_sigma_x = sigma_x ** 2
    sqr_sigma_y = sigma_y ** 2
    half_length = length / 2

    centre_x = kernel_size // 2
    centre_y = kernel_size // 2

    # line: A*x + B*y + C = 0
    # namely，y = tan(theta) * x
    A = np.sin(theta / 180 * np.pi)
    B = - np.cos(theta / 180 * np.pi)
    C = - (A * centre_x + B * centre_y)
    # B is used to assist in determining the left or right side
    B_dis = 10
    Bx = centre_x + np.cos((theta + 90) / 180 * np.pi) * B_dis
    By = centre_y + np.sin((theta + 90) / 180 * np.pi) * B_dis

    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            # Compute the point's distance from the line
            # d = (A * x0 + B * y0 + C) / sqrt(A**2 + B**2)
            dis_to_line = np.abs((A * i + B * j + C) / np.sqrt(A ** 2 + B ** 2))
            # Obtain its intersection with the line
            # x = (B**2 * x0 - A * B * y0 - A * C) / (A**2 + B**2)
            # y = (- A * B * x0 + A**2 * y0 - B * C) / (A**2 + B**2)
            cp_x = (B ** 2 * i - A * B * j - A * C) / (A ** 2 + B ** 2)
            cp_y = (- A * B * i + A ** 2 * j - B * C) / (A ** 2 + B ** 2)
            # 对每个点，计算 (其与直线的交点，直线中心点) 间距离
            # For each point, copmute the distance between its intersection with the line and the center of the line
            dis_to_line_centre = np.sqrt((cp_x - centre_x) ** 2 + (cp_y - centre_y) ** 2)
            # 对每个点，计算 (当前位置，直线中心点) 间距离
            # For each point, copmute the distance between it and the center of the line
            p_to_line_centre = np.sqrt((i - centre_x) ** 2 + (j - centre_y) ** 2)

            # Now  obtain the values of the two-dimensional elliptical Gaussian function
            # 将该点到直线的距离赋值给 y
            # 将该点到直线中心的距离赋值给 x
            x = dis_to_line_centre
            y = dis_to_line
            if p_to_line_centre <= half_length:
                val = np.exp(- (x ** 2 / (2 * sqr_sigma_x) + y ** 2 / (2 * sqr_sigma_y)))
                kernel[i, j] = val
                if isLeft_line(centre_x, centre_y, Bx, By, i, j):
                    kernel_l_part[i, j] = val
                    kernel_dis[i, j] = p_to_line_centre
                if isRight_line(centre_x, centre_y, Bx, By, i, j):
                    kernel_r_part[i, j] = val
                    kernel_dis[i, j] = - p_to_line_centre
            else:
                kernel[i, j] = 0
                kernel_l_part[i, j] = 0
                kernel_r_part[i, j] = 0

    max_val = kernel.max()
    kernel = discard_tiny_weight(kernel, max_val, 0.3)
    kernel_l_part = discard_tiny_weight(kernel_l_part, max_val, 0.3)
    kernel_r_part = discard_tiny_weight(kernel_r_part, max_val, 0.3)
    kernel = kernel / max_val
    kernel_l_part = kernel_l_part / max_val
    kernel_r_part = kernel_r_part / max_val
    return kernel, kernel_l_part, kernel_r_part, kernel_dis
def LP_line(num_line, length, k_thresh, lr_thresh):
    LP = []
    # key parameters
    delta_degree = 180 // num_line
    sigma_x = 10.0
    sigma_y = 1.0
    length = length
    k_thresh = k_thresh
    lr_thresh = lr_thresh
    length_faci = length + 4

    for n_idx in range(num_line):
        theta = n_idx * delta_degree
        temp_LP = LocalPattern()

        kernel, kernel_l_part, kernel_r_part, kernel_dis = line_kernel(sigma_x, sigma_y, length, theta)
        ksize = kernel.shape[0]

        # k   k_l   k_r
        temp_LP.k = np.zeros([ksize, ksize, num_line], dtype=np.float32)
        temp_LP.k_l = np.copy(temp_LP.k)
        temp_LP.k_r = np.copy(temp_LP.k)
        temp_LP.k[:, :, n_idx] = kernel
        temp_LP.k_l[:, :, n_idx] = kernel_l_part
        temp_LP.k_r[:, :, n_idx] = kernel_r_part

        # k_f
        temp_1, _, _, _ = line_kernel(sigma_x, sigma_y, length_faci, theta)
        temp_1_ksize = temp_1.shape[0]
        temp_2 = pad_kernel(kernel, temp_1_ksize)
        faci_kernel = temp_1 - temp_2
        temp_LP.k_f = np.zeros([temp_1_ksize, temp_1_ksize, num_line], dtype=np.float32)
        temp_LP.k_f[:, :, n_idx] = faci_kernel

        # Normalize to a common value range
        # 将模式核，模式核左部分，模式核右部分，模式促进核，全放在一个尺度下
        sum_val = temp_LP.k.sum()
        temp_LP.k /= sum_val
        temp_LP.k_l /= sum_val
        temp_LP.k_r /= sum_val
        temp_LP.k_f /= sum_val

        temp_LP.k_thresh = k_thresh    # 该值是累加局部模式 一半连续长度 的权重所得
        temp_LP.lr_thresh = lr_thresh  # 该值是累加局部模式的左或右半部分的 一半连续长度 的权重所得

        # l_link_mask
        temp_LP.l_link_mask = np.zeros([ksize, ksize], dtype=np.float32)
        t = np.copy(kernel_l_part)
        t[t != 0] = 1
        temp_LP.l_link_mask[:, :] = t

        # r_link_mask
        temp_LP.r_link_mask = np.zeros([ksize, ksize], dtype=np.float32)
        t = np.copy(kernel_r_part)
        t[t != 0] = 1
        temp_LP.r_link_mask[:, :] = t

        LP.append(temp_LP)
    return LP
