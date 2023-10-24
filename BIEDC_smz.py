import numpy as np
import cv2
import os
import time


from utils import circular_avg_kernel, gaussian_gradient_kernel
from utils import zero_kernel_centre, ellipse_gaussian_kernel, late_inhi_kernel, texture_gradient_kernel
from utils import LP_line
from utils import get_on_off, get_local_max, Relu_thresh
from utils import pad_tensor_2d
from utils import find_key_point, dda
from utils import ContinuousEdge, draw_continuous_edge
from utils import rm_mkdir_my, visualize_png


def compute_local_contrast(gray_img, radius):
    """
    compute_local_contrast(ndarray, dtype=float)

    Get the local contrast of the grayscale image.

    Parameters
    ----------
    gray_img: ndarray
        Grayscale image.
    radius:
        Area radius for computing the local contrast.

    Returns
    -------
    out: ndarray
        A two_dimension array representing the local contrast of the grayscale image.
    """

    avg_kernel = circular_avg_kernel(radius)
    mean_img = cv2.filter2D(gray_img, ddepth=-1, kernel=avg_kernel, borderType=cv2.BORDER_REFLECT)
    variance_img = np.power(gray_img - mean_img, 2)
    mean_variance_img = cv2.filter2D(variance_img, ddepth=-1, kernel=avg_kernel, borderType=cv2.BORDER_REFLECT)
    std_deviation_img = np.sqrt(mean_variance_img)
    local_contrast = np.sqrt(std_deviation_img)
    return local_contrast
def Retina(input_img):
    """
    Retina(ndarray, dtype=float)

    Model the retina to code multi information channels.

    Parameters
    ----------
    input_img: ndarray
         Input image.

    Returns
    -------
    out: ndarray
        The coded information channels in the retina, including rg, im_rg, by, im_by, luminance, and luminance contrast.
        A two_dimension array representing the local contrast of the grayscale image.
    """

    # preprocessing
    sqrt_img = np.sqrt(input_img)

    # channels: rg, im_rg, by, im_by, luminance, luminance contrast
    channels = np.zeros([input_img.shape[0], input_img.shape[1], 6], dtype=np.float32)
    channels[:, :, 0] = sqrt_img[:, :, 2] - sqrt_img[:, :, 1]
    channels[:, :, 1] = sqrt_img[:, :, 0] - 0.5 * (sqrt_img[:, :, 2] + sqrt_img[:, :, 1])
    channels[:, :, 2] = sqrt_img[:, :, 2] - 0.5 * sqrt_img[:, :, 1]
    channels[:, :, 3] = sqrt_img[:, :, 0] - 0.5 * 0.5 * (sqrt_img[:, :, 2] + sqrt_img[:, :, 1])
    lu = 0.114 * sqrt_img[:, :, 0] + 0.587 * sqrt_img[:, :, 1] + 0.299 * sqrt_img[:, :, 2]
    channels[:, :, 4] = lu
    lc = compute_local_contrast(lu, radius=2.5)
    channels[:, :, 5] = lc
    return channels


def surround_modulation(orientation_edge):
    """
    surround_modulation(ndarray)

    Design a new surround modulation mechanism to modulate the response of the classical receptive field.

    Parameters
    ----------
    orientation_edge: ndarray
        A ndarray with shape (H, W, num_orientation).

    Returns
    -------
    out: ndarray
        A ndarray with shape (H, W, num_orientation).
    """

    num_orientation = orientation_edge.shape[2]

    same_faci_kernels = []
    late_inhi_kernels = []
    for i in range(num_orientation):
        sigma_x = 0.7
        sigma_y = 2.0
        same_faci_k = ellipse_gaussian_kernel(sigma_x, sigma_y, i / num_orientation * np.pi)
        same_faci_k = zero_kernel_centre(same_faci_k, 1.5)
        same_faci_kernels.append(same_faci_k)

        sigma_x = 0.7
        sigma_y = 2.0
        late_inhi_k = late_inhi_kernel(sigma_x, sigma_y, 15, i / num_orientation * np.pi)
        late_inhi_k = zero_kernel_centre(late_inhi_k, 1.5)
        late_inhi_kernels.append(late_inhi_k)

    sum_edge = np.sum(orientation_edge, axis=2)
    avg_edge = sum_edge / num_orientation
    full_inhi_radius = same_faci_kernels[0].shape[0] // 2
    full_inhi_k = circular_avg_kernel(full_inhi_radius)
    full_inhi_k = zero_kernel_centre(full_inhi_k, centre_radius=1.5)
    full_inhi = cv2.filter2D(avg_edge, ddepth=-1, kernel=full_inhi_k, borderType=cv2.BORDER_CONSTANT)
    orientation_edge_sm = np.zeros(orientation_edge.shape, dtype=np.float32)
    for i in range(num_orientation):
        d_edge = orientation_edge[:, :, i]
        same_faci_k = same_faci_kernels[i]
        same_faci = cv2.filter2D(d_edge, ddepth=-1, kernel=same_faci_k)
        late_inhi_k = late_inhi_kernels[i]
        late_inhi = cv2.filter2D(d_edge, ddepth=-1, kernel=late_inhi_k)

        temp = d_edge + same_faci - 1.0 * late_inhi - 0.8 * full_inhi
        temp[temp < 0] = 0
        orientation_edge_sm[:, :, i] = temp

    return orientation_edge_sm
def compute_texture_boundary(texture_info, num_orientation):
    """
        compute_texture_boundary(ndarray, dtype=int)

        Compute the texture boundaries by applying the texture gradient to the texture response.

        Parameters
        ----------
        texture_info: ndarray
            Texture response of image with the shape (H X W).
        num_orientation: int
            The number of edge orientation.

        Returns
        -------
        out: ndarray
            A three_dimension array (H X W X N) representing the texture boundaries with different directions.
    """

    radiuses = [3.5, 5.5, 8.5, 12.5, 17.5]  # the radius of texture gradient kernel

    num_directions = 2 * num_orientation  # the number of the texture boundary direction
    texture_boundary = np.zeros([texture_info.shape[0], texture_info.shape[1], num_directions], dtype=np.float32)
    for i, radius in enumerate(radiuses):

        texture_boundary_24s = np.zeros([texture_info.shape[0], texture_info.shape[1], num_directions], dtype=np.float32)
        for idx in range(num_directions):
            theta = idx * 15
            texture_boundary_k = texture_gradient_kernel(theta, radius)
            temp = cv2.filter2D(texture_info, ddepth=-1, kernel=texture_boundary_k, borderType=cv2.BORDER_REFLECT)

            dis = - texture_boundary_k.shape[0] // 2 // 2
            M = np.float32([[1, 0, dis * np.cos(theta / 180 * np.pi + np.pi / 2)],
                            [0, 1, dis * np.sin(theta / 180 * np.pi + np.pi / 2)]])
            texture_boundary_24s[:, :, idx] = cv2.warpAffine(temp, M, (temp.shape[1], temp.shape[0]))

        texture_boundary[:, :, :] += texture_boundary_24s

    texture_boundary[texture_boundary < 0] = 0
    return texture_boundary
def fuse_sc_with_tb(crf_response):
    """
        fuse_sc_with_tb(ndarray)

        Fuse the response of simple cells with the texture boundaries.
        This function first applies surround modulation mechanism to modulate the response of simple ccells.
        Then get texture information by summing the classical receptive's response of simple cells
        and apply texture gradient to get texture boundaries.
        The surround modulation mechanism is also used to the texture boundaries.
        At last, fuse the response of simple cells and the texture boundaries.

        Parameters
        ----------
        crf_reponse: ndarray
            The classical receptive field respopns of the simple cells with the shape H X W X N.

        Returns
        -------
        out: ndarray
            A two_dimension array (H X W) response the result of the fusion.
    """

    rf_response = surround_modulation(crf_response)
    sc_edge = np.max(rf_response, axis=2)
    max_val = sc_edge.max()
    sc_edge[sc_edge > max_val / 7] = max_val / 7
    sc_edge /= sc_edge.max()

    texture_info = np.sum(crf_response, axis=2)
    texture_boundary_24 = compute_texture_boundary(texture_info, crf_response.shape[2])
    texture_boundary_so = np.zeros(crf_response.shape, dtype=np.float32)  # texture boundary with the same orientation
    for i in range(12):
        texture_boundary_so[:, :, i] = texture_boundary_24[:, :, i] + texture_boundary_24[:, :, i+12]
    fuse_edge_sm = surround_modulation(texture_boundary_so)
    texture_boundary = np.max(fuse_edge_sm, axis=2)
    max_val = texture_boundary.max()
    texture_boundary[texture_boundary > max_val / 7] = max_val / 7
    texture_boundary /= texture_boundary.max()

    edge = sc_edge * texture_boundary
    return edge
def V1_onescale(chs, num_orientation):
    """
        V1_onescale(ndarray, dtype=int)

        Get the output of V1 at one image scale.

        Parameters
        ----------
        chs: ndarray
            Coded information channels with the shape H X W X N.
        num_orientation:
            The number of the orientation edges.

        Returns
        -------
        out: ndarray
            A two_dimension array (H X W) the output of V1 at one scale.
    """
    edge_response = np.zeros([chs.shape[0], chs.shape[1], chs.shape[2]], dtype=np.float32)

    for idx_c in range(chs.shape[2]):
        channel_info = chs[:, :, idx_c]
        crf_response = np.zeros([channel_info.shape[0], channel_info.shape[1], num_orientation], dtype=np.float32)
        for idx in range(num_orientation):
            theta = idx / num_orientation * np.pi
            for sigma in [1]:
                crf_k = gaussian_gradient_kernel(sigma, theta, 0.3)
                crf = np.abs(cv2.filter2D(channel_info, ddepth=-1, kernel=crf_k, borderType=cv2.BORDER_REFLECT))
                crf_response[:, :, idx] += crf * 1 / sigma

        edge = fuse_sc_with_tb(crf_response)
        edge_response[:, :, idx_c] = edge

    edge = np.sum(edge_response, axis=2)
    # edge /= edge.max()

    return edge
def V1(input_img):
    """
        V1(ndarray)

        Get the output of V1.

        Parameters
        ----------
        input_img: ndarray
            Input image with the shape H X W X 3.

        Returns
        -------
        out: ndarray
            A two_dimension array (H X W) representing the output of V1.
    """
    h, w = input_img.shape[0], input_img.shape[1]
    result = np.zeros([h, w, 4], dtype=np.float32)
    for i in range(4):
        k = np.power(2, i)
        timg = cv2.resize(input_img, (0, 0), fx=1/k, fy=1/k, interpolation=cv2.INTER_LINEAR)
        channels = Retina(timg)
        edge = V1_onescale(channels, 12)
        edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
        result[:, :, i] = edge

    edge = result[:, :, 0] + 1/2 * result[:, :, 1] + 1/3 * result[:, :, 2] + 1/4 * result[:, :, 3]
    edge /= edge.max()
    return edge


def thin_edge(edge):
    """
        thin_edge(ndarray)

        Thin the edge map.
        First estimate the optimal orientation of edge.
        Then apply the non-maximum suppression along the orthogonal orientation to thin edge.

        Parameters
        ----------
        edge: ndarray
            The edge map with the shape H X W.

        Returns
        -------
        out: ndarray
            A two_dimension array (H X W) representing  the result of thinning.
    """
    # 这个函数的思路是这样的
    # 估计当前位置的最优边缘朝向，然后在边缘朝向的垂直方向做抑制

    degree = 15
    kk = 3

    # 特别注意，我输入的 theta，其竖直方向为 0 度，且逆时针方向旋转
    # 而 for 循环中的 theta，其竖直方向为 0 度，但是为顺时针旋转
    # 这二者的不同导致最后进行细化(非最大值抑制)时 theta 需要一致，不然结果不对
    # 中心点，左右邻居
    top_kernels = []
    bottom_kernels = []
    for n in range(12):
        top_k = np.zeros([3, 3], dtype=np.float32)
        bottom_k = np.zeros([3, 3], dtype=np.float32)
        theta = (degree * n / 180 * np.pi) % np.pi
        # [0, 45)    [180, 225)
        if 0 <= n < 1 * kk or 4 * kk <= n < 5 * kk:
            top_k[2, 2] = np.tan(theta)
            bottom_k[0, 0] = np.tan(theta)
            top_k[1, 2] = 1 - np.tan(theta)
            bottom_k[1, 0] = 1 - np.tan(theta)
        # [45, 90)    [225, 270)
        elif 1 * kk <= n < 2 * kk or 5 * kk <= n < 6 * kk:
            theta = np.pi / 2 - theta
            top_k[2, 2] = np.tan(theta)
            bottom_k[0, 0] = np.tan(theta)
            top_k[2, 1] = 1 - np.tan(theta)
            bottom_k[0, 1] = 1 - np.tan(theta)
        # [90, 135)    [270, 315)
        elif 2 * kk <= n < 3 * kk or 6 * kk <= n < 7 * kk:
            theta = theta - np.pi / 2
            top_k[2, 0] = np.tan(theta)
            bottom_k[0, 2] = np.tan(theta)
            top_k[2, 1] = 1 - np.tan(theta)
            bottom_k[0, 1] = 1 - np.tan(theta)
        # [135, 180)   [315, 360)
        else:
            theta = np.pi - theta
            top_k[2, 0] = np.tan(theta)
            bottom_k[0, 2] = np.tan(theta)
            top_k[1, 0] = 1 - np.tan(theta)
            bottom_k[1, 2] = 1 - np.tan(theta)

        top_kernels.append(top_k)
        bottom_kernels.append(bottom_k)

    temp_edge_3 = np.zeros([edge.shape[0], edge.shape[1], 12, 3], dtype=np.float32)
    orient_estimation = np.zeros([edge.shape[0], edge.shape[1], 12], dtype=np.float32)
    for n in range(12):
        top_k = top_kernels[n]
        bottom_k = bottom_kernels[n]
        temp_edge_3[:, :, n, 0] = cv2.filter2D(edge[:, :], ddepth=-1, kernel=top_k)
        temp_edge_3[:, :, n, 1] = edge[:, :]
        temp_edge_3[:, :, n, 2] = cv2.filter2D(edge[:, :], ddepth=-1, kernel=bottom_k)

        k = ellipse_gaussian_kernel(1.5, 4, n * degree / 180 * np.pi)
        orient_estimation[:, :, n] = cv2.filter2D(edge, ddepth=-1, kernel=k)

    # 找到预估的边缘朝向
    best_orient_estimation = np.argmax(orient_estimation, axis=2)

    result = np.zeros(edge.shape, dtype=np.float32)
    for h in range(edge.shape[0]):
        for w in range(edge.shape[1]):
            # 获得预估朝向的对应的垂直方向
            best_orien = best_orient_estimation[h, w]  # 预估边缘最优朝向
            # 对应的垂直朝向
            orien = (best_orien + 6) % 12
            if temp_edge_3[h, w, orien, 1] >= temp_edge_3[h, w, orien, 0] and temp_edge_3[h, w, orien, 1] >= temp_edge_3[h, w, orien, 2]:
                result[h, w] = edge[h, w]
            else:
                result[h, w] = 0

    return result
def V2(V1_edge, LPs):
    """
        V2(ndarray, dtype=[])

        Get the response of orientation-sensitive cells, left endpoint cells, and right endpoint cells of V2.

        Parameters
        ----------
        V1_edge: ndarray
            The output of V1.
        LPs:
            The data structure provided the

        Returns
        -------
        out: ndarray
            A two_dimension array representing the local contrast of the grayscale image.
    """

    # SC 局部模式数量
    num_pattern = len(LPs)
    shape = V1_edge.shape
    SC_response = np.zeros([shape[0], shape[1], num_pattern], dtype=np.float32)
    HC_l_response = np.zeros([shape[0], shape[1], num_pattern], dtype=np.float32)
    HC_r_response = np.zeros([shape[0], shape[1], num_pattern], dtype=np.float32)

    inter_neuron = get_local_max(V1_edge, 11) + 1e-4

    for n_idx in range(num_pattern):
        # 注意：模式核中所有权重之和为 1，左半部分，右半部分，促进核，全都统一到了一个权重尺度
        P = LPs[n_idx]

        # 经典感受野响应
        k = np.max(P.k, axis=2)
        p_edge = cv2.filter2D(V1_edge, ddepth=-1, kernel=k, borderType=cv2.BORDER_CONSTANT)

        # 每个位置有值，不代表该位置存在某种局部边缘模式，该值必须符合一定条件，才能代表该局部边缘模式存在
        # 此处可以视作中间神经元收集局部范围内局部边缘模式的标准值
        # 若当前位置与中间神经元的比值达到一定比例，我们认为此处存在局部边缘模式
        thresh = P.k_thresh
        on_off = get_on_off(p_edge, inter_neuron, thresh)

        # 利用中间神经元去掉那些不存在局部边缘的地方
        p_edge_on_off = p_edge * on_off

        # 然后，引入周围调节
        k_faci = np.max(P.k_f, axis=2)
        faci = cv2.filter2D(V1_edge, ddepth=-1, kernel=k_faci, borderType=cv2.BORDER_CONSTANT)
        # 这里，边缘首先存在，然后周围调节才是有意义的
        p_edge_faci = (p_edge_on_off + faci) * on_off

        SC_response[:, :, n_idx] = p_edge_faci

        # 获得模式的左右部分响应
        k_l = np.max(P.k_l, axis=2)
        p_edge_l = cv2.filter2D(V1_edge, ddepth=-1, kernel=k_l, borderType=cv2.BORDER_CONSTANT)
        k_r = np.max(P.k_r, axis=2)
        p_edge_r = cv2.filter2D(V1_edge, ddepth=-1, kernel=k_r, borderType=cv2.BORDER_CONSTANT)
        thresh = P.lr_thresh
        p_edge_l_on_off = get_on_off(p_edge_l, inter_neuron, thresh=thresh)
        p_edge_r_on_off = get_on_off(p_edge_r, inter_neuron, thresh=thresh)

        HC_l_response[:, :, n_idx] = p_edge_l_on_off
        HC_r_response[:, :, n_idx] = p_edge_r_on_off

    # 归一化
    SC_response /= SC_response.max() + 1e-4
    # 小于一定阈值的边缘，神经元是不会发放的
    SC_response = Relu_thresh(SC_response, thresh=0.1 * SC_response.max())

    return SC_response, HC_l_response, HC_r_response
def connect_edge(V2_SC_response, V2_HC_l_response, V2_HC_r_response, V2_LP):
    """
        connect_edge(ndarray, ndarray, ndarray, dtype=[])

        Connect edges by utilizing the endpoint cells.

        Parameters
        ----------
        : ndarray
            Grayscale image.
        radius:
            Area radius for computing the local contrast.

        Returns
        -------
        out: ndarray
            A two_dimension array representing the local contrast of the grayscale image.
    """

    # 在每个位置，获得局部模式响应的最大值及其对应的标号
    p_argmax = np.argmax(V2_SC_response, axis=2)
    p_max = np.max(V2_SC_response, axis=2)

    CE = []
    # 若某位置已经被表达过，则记录，从而避免大量表达重叠
    mask = np.zeros(p_max.shape, dtype=int)

    # 用于避免超出图像大小范围
    padding_half_size = 11
    temp_p_max = pad_tensor_2d(p_max, padding_half_size)

    # 对每个位置的值，来考虑是否该处存在一个局部边缘结构
    h = V2_SC_response.shape[0]
    w = V2_SC_response.shape[1]
    for i in range(h):
        for j in range(w):
            # 获得每个位置上的值
            val = p_max[i, j]

            # 当不存在边缘结构时，不需要考虑
            if val == 0:
                continue
            # 当该位置已经被连接过时，不需要考虑
            if mask[i, j] == 1:
                continue

            # [i, j] 实际上为该边缘的中心，同时更新模
            mask[i, j] = 1

            # 迭代判断判断左右端点是否存在

            # 考虑左侧
            edge_L_part = []
            cur_x = i
            cur_y = j
            # 获得当前边缘结构标号
            argmax_o = p_argmax[i, j]
            # 获得左边 mask
            l_link_mask = V2_LP[argmax_o].l_link_mask
            L_being = V2_HC_l_response[cur_x, cur_y, argmax_o]
            # 该点左侧存在
            while L_being == 1:
                # 从近到远添加左侧边缘点
                # 首先，利用 l_link_mask 寻找连续的关键点
                # 获得局部区域
                mask_half_size = l_link_mask.shape[0] // 2
                t_i_b = padding_half_size + cur_x - mask_half_size
                t_i_u = padding_half_size + cur_x + mask_half_size + 1
                t_j_b = padding_half_size + cur_y - mask_half_size
                t_j_u = padding_half_size + cur_y + mask_half_size + 1
                local_area = temp_p_max[t_i_b:t_i_u, t_j_b:t_j_u]
                left_point, local_max_val = find_key_point(local_area, l_link_mask)
                # 如果左侧不存在边缘了，则停止循环(理论上不该出现这种情况)
                if local_max_val == 0:
                    break
                # 找到左侧边缘点位置
                l_pos = np.zeros([2], dtype=int)
                l_pos[0] = cur_x + left_point[0] - mask_half_size
                l_pos[1] = cur_y + left_point[1] - mask_half_size
                # 获得左侧边缘结构，注意返回的边缘是包含起点与端点的完整边缘
                edge = dda(cur_x, cur_y, l_pos[0], l_pos[1])
                # 移除起点，这样做的主要原因在于，我们在前面已经将 起点 的 mask 置为 1 了
                del edge[0]
                # 如果左侧已经连接过了，则停止循环
                # 这里有个问题，有可能会穿过已有边缘，思路是相加这些点的 mask，只要不为 0 即穿过，则停止
                # 注意，理论上存在一种极其特别的情况，即 45 度交叉线条，难以考虑
                flag = 0
                for e in edge:
                    flag += mask[e[0], e[1]]
                if flag != 0:
                    break
                # e = edge[-1]
                if mask[e[0], e[1]]:
                    break
                # 否则左侧存在边缘，则找到关键点对应图中的位置，加入该位置
                # 值得注意的是，坐标永远不会超出图像的范围，原因在于 padding 的值为 0
                else:
                    # 左点加入了当前的局部边缘结构
                    edge_L_part += edge
                    # 更新模
                    for e in edge:
                        mask[e[0], e[1]] = 1
                    # 更新循环所需信息
                    cur_x = l_pos[0]
                    cur_y = l_pos[1]
                    argmax_o = p_argmax[cur_x, cur_y]
                    l_link_mask = V2_LP[argmax_o].l_link_mask
                    L_being = V2_HC_l_response[cur_x, cur_y, argmax_o]

            # 考虑右侧
            edge_R_part = []
            cur_x = i
            cur_y = j
            # 获得当前边缘结构标号
            argmax_o = p_argmax[i, j]
            # 获得右边 mask
            r_link_mask = V2_LP[argmax_o].r_link_mask
            R_being = V2_HC_r_response[cur_x, cur_y, argmax_o]
            # 该点右侧存在
            while R_being == 1:
                # 从近到远添加右侧边缘点
                # 首先，利用 r_link_mask 寻找连续的关键点
                # 获得局部区域
                mask_half_size = r_link_mask.shape[0] // 2
                t_i_b = padding_half_size + cur_x - mask_half_size
                t_i_u = padding_half_size + cur_x + mask_half_size + 1
                t_j_b = padding_half_size + cur_y - mask_half_size
                t_j_u = padding_half_size + cur_y + mask_half_size + 1
                local_area = temp_p_max[t_i_b:t_i_u, t_j_b:t_j_u]
                right_point, local_max_val = find_key_point(local_area, r_link_mask)
                # 如果右侧不存在边缘了，则停止循环(理论上不该出现这种情况)
                if local_max_val == 0:
                    break
                # 找到右侧边缘点位置
                r_pos = np.zeros([2], dtype=int)
                r_pos[0] = cur_x + right_point[0] - mask_half_size
                r_pos[1] = cur_y + right_point[1] - mask_half_size
                # 获得右侧边缘结构，注意返回的边缘是包含起点与端点的完整边缘
                edge = dda(cur_x, cur_y, r_pos[0], r_pos[1])
                # 移除起点，这样做的主要原因在于，我们在前面已经将 起点 的 mask 置为 1 了
                del edge[0]
                # 如果右侧已经连接过了，则停止循环
                # 这里有个问题，有可能会穿过已有边缘，思路是相加这些点的 mask，只要不为 0 即穿过，则停止
                # 注意，理论上存在一种极其特别的情况，即 45 度交叉线条，难以考虑
                flag = 0
                for e in edge:
                    flag += mask[e[0], e[1]]
                if flag != 0:
                    break
                # e = edge[-1]
                if mask[e[0], e[1]]:
                    break
                # 否则右侧存在边缘，则找到关键点对应图中的位置，加入该位置
                # 值得注意的是，坐标永远不会超出图像的范围，原因在于 padding 的值为 0
                else:
                    # 右点加入了当前的局部边缘结构
                    edge_R_part += edge
                    # 更新模
                    for e in edge:
                        mask[e[0], e[1]] = 1
                    # 更新循环所需信息
                    cur_x = r_pos[0]
                    cur_y = r_pos[1]
                    argmax_o = p_argmax[cur_x, cur_y]
                    r_link_mask = V2_LP[argmax_o].r_link_mask
                    R_being = V2_HC_r_response[cur_x, cur_y, argmax_o]

            # 拼成一个连续的线
            # 左侧
            edge_L_part.reverse()
            merged_edge = edge_L_part
            # 中间点
            merged_edge.append([i, j])
            # 右侧
            merged_edge += edge_R_part
            # 现在去除重复点
            complete_edge = []
            merged_length = len(merged_edge)
            complete_edge.append(merged_edge[0])
            for idx in range(1, merged_length):
                if complete_edge[-1] == merged_edge[idx]:
                    continue
                else:
                    complete_edge.append(merged_edge[idx])

            ce = ContinuousEdge()
            ce.key_points = complete_edge
            ce.pattern_idx = argmax_o  # 事实上，此处非朝向边缘，这个数据段无用
            # 计算边缘的平均值
            sum_val = 0
            max_val = 0
            for e in ce.key_points:
                # sum_val += p_max[e[0], e[1]]
                if p_max[e[0], e[1]] > max_val:
                    max_val = p_max[e[0], e[1]]
            # ce.val = sum_val / len(ce.key_points)
            ce.val = max_val
            CE.append(ce)

    return CE


def run_model(img_path, visu_pngs_dir):
    """
        compute_local_contrast(ndarray, dtype=float)

        Get the local contrast of the grayscale image.

        Parameters
        ----------
        gray_img: ndarray
            Grayscale image.
        radius:
            Area radius for computing the local contrast.

        Returns
        -------
        out: ndarray
            A two_dimension array representing the local contrast of the grayscale image.
    """

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_img = img / img.max()
    visualize_png(input_img, "0_img", visu_pngs_dir)

    V1_edge = V1(input_img)
    visualize_png(V1_edge, "V1_edge", visu_pngs_dir)

    edge_nms = thin_edge(V1_edge)
    visualize_png(edge_nms, "edge_nms", visu_pngs_dir)

    V2_LPs = LP_line(num_line=12, length=11, k_thresh=0.20, lr_thresh=0.10)
    V2SC_response, V2SC_l_response, V2SC_r_response = V2(edge_nms, V2_LPs)

    V2_edge = np.max(V2SC_response, axis=2)
    visualize_png(V2_edge, "V2_edge", visu_pngs_dir)

    CS = connect_edge(V2SC_response, V2SC_l_response, V2SC_r_response, V2_LPs)
    link_edge, link_edge_coloring, edge_and_img = draw_continuous_edge(CS, input_img)
    visualize_png(link_edge, "link_edge", visu_pngs_dir)
    visualize_png(link_edge_coloring, "link_edge_coloring", visu_pngs_dir)
    visualize_png(edge_and_img, "link_edge_and_img", visu_pngs_dir)

    link_edge *= 1.2
    visualize_png(link_edge, "link_edge_gussian", visu_pngs_dir)
    edge = np.maximum(V1_edge, link_edge)

    edge /= edge.max()
    visualize_png(edge, "V1_edge_w_feedback", visu_pngs_dir)
    return edge

class MyConfig():
    def __init__(self):
        self.pics_dir = os.path.join(os.path.abspath(os.curdir), 'pictures')
        img_dir_name = "MBBD100"
        self.data_dir = os.path.join(self.pics_dir, img_dir_name)

        self.result_dir = os.path.join(self.pics_dir, img_dir_name + "_sm_with_cz_result")
        self.visu_dir = os.path.join(self.pics_dir, img_dir_name + "_sm_with_cz_visualization")

        rm_mkdir_my(self.result_dir)
        rm_mkdir_my(self.visu_dir)


myconfig = MyConfig()
if __name__ == "__main__":
    start_time = time.time()
    test_pics_list = os.listdir(myconfig.data_dir)

    for idx, pic_name in enumerate(test_pics_list):
        temp_time = time.time()
        pos_suffix = pic_name.find(".")
        pic_name_wo_suffix = pic_name[:pos_suffix]
        visu_pngs_dir = os.path.join(myconfig.visu_dir, pic_name_wo_suffix)
        os.mkdir(visu_pngs_dir)

        # 获得当前处理图片的路径
        pic_path = os.path.join(myconfig.data_dir, pic_name)
        # 运行模型
        edge = run_model(pic_path, visu_pngs_dir)
        visualize_png(edge, pic_name_wo_suffix, myconfig.result_dir)

        print("Complete {:.4f}%. Cost {:.4f}s. filename: {}".format((idx + 1) / len(test_pics_list), time.time() - temp_time, pic_name))

    print("Totally cost {:.4f}s.".format(time.time() - start_time))