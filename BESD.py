import numpy as np
import cv2
import os
import time


from utils import circular_avg_kernel, gaussian_gradient_kernel
from utils import zero_kernel_centre, ellipse_gaussian_kernel, late_inhi_kernel, texture_gradient_kernel
from utils import LP_line
from utils import get_on_off, get_local_max, Relu_thresh
from utils import pad_tensor_2d
from utils import find_key_point, DDA
from utils import EdgeSegment, draw_continuous_edge
from utils import visualize_png


def compute_local_contrast(gray_img, radius):
    # Get the local contrast of the illuminance
    avg_kernel = circular_avg_kernel(radius)
    mean_img = cv2.filter2D(gray_img, ddepth=-1, kernel=avg_kernel, borderType=cv2.BORDER_REFLECT)
    variance_img = np.power(gray_img - mean_img, 2)
    mean_variance_img = cv2.filter2D(variance_img, ddepth=-1, kernel=avg_kernel, borderType=cv2.BORDER_REFLECT)
    std_deviation_img = np.sqrt(mean_variance_img)
    local_contrast = np.sqrt(std_deviation_img)
    return local_contrast
def Retina(input_img):
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
    # The surround modulation mechanism consists of same-orientation facilitation,
    # lateral inhibition, and full-orientation inhibition
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
    # Compute the texture boundaries by applying the texture gradient to the texture information.
    num_directions = 2 * num_orientation  # The number of the texture boundary direction
    radiuses = [3.5, 5.5, 8.5, 12.5, 17.5]  # The radius of texture gradient kernel
    texture_boundary = np.zeros([texture_info.shape[0], texture_info.shape[1], num_directions], dtype=np.float32)
    for i, radius in enumerate(radiuses):
        texture_boundary_24s = np.zeros([texture_info.shape[0], texture_info.shape[1], num_directions], dtype=np.float32)
        for idx in range(num_directions):
            theta = idx * 15
            texture_boundary_k = texture_gradient_kernel(theta, radius)
            temp = cv2.filter2D(texture_info, ddepth=-1, kernel=texture_boundary_k, borderType=cv2.BORDER_REFLECT)

            # Translate it to address the double detection problem
            dis = - texture_boundary_k.shape[0] // 2 // 2
            M = np.float32([[1, 0, dis * np.cos(theta / 180 * np.pi + np.pi / 2)],
                            [0, 1, dis * np.sin(theta / 180 * np.pi + np.pi / 2)]])
            texture_boundary_24s[:, :, idx] = cv2.warpAffine(temp, M, (temp.shape[1], temp.shape[0]))

        texture_boundary[:, :, :] += texture_boundary_24s

    texture_boundary[texture_boundary < 0] = 0
    return texture_boundary
def fuse_sc_with_tb(crf_response):
    """
        Fuse the response of simple cells with the texture boundaries.
        This function first applies surround modulation mechanism to modulate the response of simple ccells.
        Then get texture information by summing the classical receptive's response of simple cells
        and apply texture gradient to get texture boundaries.
        The surround modulation mechanism is also used to the texture boundaries.
        At last, fuse the response of simple cells and the texture boundaries.
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
    #Get the output of V1 at one image scale.
    edge_response = np.zeros([chs.shape[0], chs.shape[1], chs.shape[2]], dtype=np.float32)
    for idx_c in range(chs.shape[2]):
        channel_info = chs[:, :, idx_c]
        crf_response = np.zeros([channel_info.shape[0], channel_info.shape[1], num_orientation], dtype=np.float32)
        for idx in range(num_orientation):
            theta = idx / num_orientation * np.pi
            crf_k = gaussian_gradient_kernel(1, theta, 0.3)
            crf = np.abs(cv2.filter2D(channel_info, ddepth=-1, kernel=crf_k, borderType=cv2.BORDER_REFLECT))
            crf_response[:, :, idx] += crf
        # Fuse the response of simple cells with the texture boundaries.
        edge = fuse_sc_with_tb(crf_response)
        edge_response[:, :, idx_c] = edge

    # Combine all channels' results
    edge = np.sum(edge_response, axis=2)
    return edge
def Retina_V1(input_img):
    # Get the output of V1.
    h, w = input_img.shape[0], input_img.shape[1]
    result = np.zeros([h, w, 4], dtype=np.float32)
    for i in range(4):
        k = np.power(2, i)
        timg = cv2.resize(input_img, (0, 0), fx=1/k, fy=1/k, interpolation=cv2.INTER_LINEAR)
        channels = Retina(timg)
        edge = V1_onescale(channels, 12)
        edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
        result[:, :, i] = edge
    # Combine the results of all scale
    edge = result[:, :, 0] + 1/2 * result[:, :, 1] + 1/3 * result[:, :, 2] + 1/4 * result[:, :, 3]
    edge /= edge.max()
    return edge


def thin_edge(V1_edge):
    """
        Thin the edge map.
        First estimate the optimal orientation of edge.
        Then apply the non-maximum suppression along the orthogonal orientation to thin edge.
    """
    degree = 15
    kk = 3
    # 特别注意，我输入的 theta，其竖直方向为 0 度，且逆时针方向旋转
    # 而 for 循环中的 theta，其竖直方向为 0 度，但是为顺时针旋转
    # 这二者的不同导致最后进行细化(非最大值抑制)时 theta 需要一致，不然结果不对
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

    temp_edge_3 = np.zeros([V1_edge.shape[0], V1_edge.shape[1], 12, 3], dtype=np.float32)
    orient_estimation = np.zeros([V1_edge.shape[0], V1_edge.shape[1], 12], dtype=np.float32)
    for n in range(12):
        top_k = top_kernels[n]
        bottom_k = bottom_kernels[n]
        temp_edge_3[:, :, n, 0] = cv2.filter2D(V1_edge[:, :], ddepth=-1, kernel=top_k)
        temp_edge_3[:, :, n, 1] = V1_edge[:, :]
        temp_edge_3[:, :, n, 2] = cv2.filter2D(V1_edge[:, :], ddepth=-1, kernel=bottom_k)

        k = ellipse_gaussian_kernel(1.5, 4, n * degree / 180 * np.pi)
        orient_estimation[:, :, n] = cv2.filter2D(V1_edge, ddepth=-1, kernel=k)

    # Estimate the optimal orientation
    best_orient_estimation = np.argmax(orient_estimation, axis=2)

    result = np.zeros(V1_edge.shape, dtype=np.float32)
    for h in range(V1_edge.shape[0]):
        for w in range(V1_edge.shape[1]):
            # Get its orthogonal orientation
            best_orien = best_orient_estimation[h, w]
            orth_orien = (best_orien + 6) % 12
            if temp_edge_3[h, w, orth_orien, 1] >= temp_edge_3[h, w, orth_orien, 0] and temp_edge_3[h, w, orth_orien, 1] >= temp_edge_3[h, w, orth_orien, 2]:
                result[h, w] = V1_edge[h, w]
            else:
                result[h, w] = 0
    return result
def V2(thinned_edge, LPs):
    # Get the response of simple cells, left end-stopping cells, and right end-stopping cells of V2.
    num_pattern = len(LPs)  # The number of local patterns
    shape = thinned_edge.shape
    SC_response = np.zeros([shape[0], shape[1], num_pattern], dtype=np.float32)
    HC_l_response = np.zeros([shape[0], shape[1], num_pattern], dtype=np.float32)
    HC_r_response = np.zeros([shape[0], shape[1], num_pattern], dtype=np.float32)
    M = get_local_max(thinned_edge, 11) + 1e-4  # The local maximum values

    for n_idx in range(num_pattern):
        P = LPs[n_idx]
        k = np.max(P.k, axis=2)
        sc = cv2.filter2D(thinned_edge, ddepth=-1, kernel=k, borderType=cv2.BORDER_CONSTANT)
        # When the ratio of sc and M reaches a certain threshold,
        # we consider the presence of a local edge pattern
        on_off = get_on_off(sc, M, P.k_thresh)
        p_edge_on_off = sc * on_off
        k_faci = np.max(P.k_f, axis=2)
        faci = cv2.filter2D(thinned_edge, ddepth=-1, kernel=k_faci, borderType=cv2.BORDER_CONSTANT)
        p_edge_faci = (p_edge_on_off + faci) * on_off
        SC_response[:, :, n_idx] = p_edge_faci

        # Obtain the response of the left end-stopping cell
        k_l = np.max(P.k_l, axis=2)
        p_edge_l = cv2.filter2D(thinned_edge, ddepth=-1, kernel=k_l, borderType=cv2.BORDER_CONSTANT)
        p_edge_l_on_off = get_on_off(p_edge_l, M, thresh=P.lr_thresh)
        # Obtain the response of the left end-stopping cell
        k_r = np.max(P.k_r, axis=2)
        p_edge_r = cv2.filter2D(thinned_edge, ddepth=-1, kernel=k_r, borderType=cv2.BORDER_CONSTANT)
        p_edge_r_on_off = get_on_off(p_edge_r, M, thresh=P.lr_thresh)
        HC_l_response[:, :, n_idx] = 1 - p_edge_l_on_off
        HC_r_response[:, :, n_idx] = 1 - p_edge_r_on_off

    SC_response /= SC_response.max() + 1e-4
    # If cells with values below a certain threshold, they do not fire
    SC_response = Relu_thresh(SC_response, thresh=0.1 * SC_response.max())
    return SC_response, HC_l_response, HC_r_response
def EdgeSegmentDetectionAlgorithm(V2_SC_response, V2_HC_l_response, V2_HC_r_response, V2_LP):
    # Connect edges by utilizing the end-stopping cells
    AO = np.argmax(V2_SC_response, axis=2)
    V = np.max(V2_SC_response, axis=2)
    ES = []
    Mask = np.zeros(V.shape, dtype=int)

    padding_half_size = 11
    temp_p_max = pad_tensor_2d(V, padding_half_size)

    h = V2_SC_response.shape[0]
    w = V2_SC_response.shape[1]
    for i in range(h):
        for j in range(w):
            val = V[i, j]
            # When it is not an edge, pass
            if val == 0:
                continue
            # When it has been processed, pass
            if Mask[i, j] == 1:
                continue
            # Flag it
            Mask[i, j] = 1

            # Connect the left side
            L_part = []
            cur_x = i
            cur_y = j
            # Obtain the optimal orientation of current position
            argmax_o = AO[i, j]
            # Obtain the left region mask for searching the next key point
            l_link_mask = V2_LP[argmax_o].l_link_mask
            L_being = V2_HC_l_response[cur_x, cur_y, argmax_o]
            while L_being == 0:
                mask_half_size = l_link_mask.shape[0] // 2
                t_i_b = padding_half_size + cur_x - mask_half_size
                t_i_u = padding_half_size + cur_x + mask_half_size + 1
                t_j_b = padding_half_size + cur_y - mask_half_size
                t_j_u = padding_half_size + cur_y + mask_half_size + 1
                local_area = temp_p_max[t_i_b:t_i_u, t_j_b:t_j_u]
                left_point, local_max_val = find_key_point(local_area, l_link_mask)
                # If there are no edges on the left, the loop stops (in theory, this situation should not occur)
                if local_max_val == 0:
                    break
                # Find the key point on the left side
                l_pos = np.zeros([2], dtype=int)
                l_pos[0] = cur_x + left_point[0] - mask_half_size
                l_pos[1] = cur_y + left_point[1] - mask_half_size
                # Apply the DDA algorithm to connect all key points
                edge = DDA(cur_x, cur_y, l_pos[0], l_pos[1])
                # Remove the start point
                del edge[0]
                # There might are edges that cross existing edges.
                # The idea is to sum the masks of these points, and if the result is not zero, the loop stops.
                # Please note this is a very rare case, such as a 45-degree crossing line.
                flag = 0
                for e in edge:
                    flag += Mask[e[0], e[1]]
                if flag != 0:
                    break
                # If the left side has already been connected, the loop stops
                if Mask[e[0], e[1]]:
                    break
                else:
                    L_part += edge
                    # Update Mask
                    for e in edge:
                        Mask[e[0], e[1]] = 1
                    # Update
                    cur_x = l_pos[0]
                    cur_y = l_pos[1]
                    argmax_o = AO[cur_x, cur_y]
                    l_link_mask = V2_LP[argmax_o].l_link_mask
                    L_being = V2_HC_l_response[cur_x, cur_y, argmax_o]

            # Connect the right side
            R_part = []
            cur_x = i
            cur_y = j
            # Obtain the optimal orientation of current position
            argmax_o = AO[i, j]
            # Obtain the right region mask for searching the next key point
            r_link_mask = V2_LP[argmax_o].r_link_mask
            R_being = V2_HC_r_response[cur_x, cur_y, argmax_o]
            while R_being == 0:
                mask_half_size = r_link_mask.shape[0] // 2
                t_i_b = padding_half_size + cur_x - mask_half_size
                t_i_u = padding_half_size + cur_x + mask_half_size + 1
                t_j_b = padding_half_size + cur_y - mask_half_size
                t_j_u = padding_half_size + cur_y + mask_half_size + 1
                local_area = temp_p_max[t_i_b:t_i_u, t_j_b:t_j_u]
                right_point, local_max_val = find_key_point(local_area, r_link_mask)
                # If there are no edges on the right, the loop stops (in theory, this situation should not occur)
                if local_max_val == 0:
                    break
                # Find the key point on the right side
                r_pos = np.zeros([2], dtype=int)
                r_pos[0] = cur_x + right_point[0] - mask_half_size
                r_pos[1] = cur_y + right_point[1] - mask_half_size
                # Apply the DDA algorithm to connect all key points
                edge = DDA(cur_x, cur_y, r_pos[0], r_pos[1])
                # Remove the start point
                del edge[0]
                # There might are edges that cross existing edges.
                # The idea is to sum the masks of these points, and if the result is not zero, the loop stops.
                # Please note this is a very rare case, such as a 45-degree crossing line.
                flag = 0
                for e in edge:
                    flag += Mask[e[0], e[1]]
                if flag != 0:
                    break
                # If the right side has already been connected, the loop stops
                if Mask[e[0], e[1]]:
                    break
                else:
                    R_part += edge
                    # Update Mask
                    for e in edge:
                        Mask[e[0], e[1]] = 1
                    # Update
                    cur_x = r_pos[0]
                    cur_y = r_pos[1]
                    argmax_o = AO[cur_x, cur_y]
                    r_link_mask = V2_LP[argmax_o].r_link_mask
                    R_being = V2_HC_r_response[cur_x, cur_y, argmax_o]

            # Integrate the left and right part
            # The left part
            L_part.reverse()
            merged_edge = L_part
            # The start point
            merged_edge.append([i, j])
            # The right part
            merged_edge += R_part
            # Remove the repeated points
            complete_edge = []
            merged_length = len(merged_edge)
            complete_edge.append(merged_edge[0])
            for idx in range(1, merged_length):
                if complete_edge[-1] == merged_edge[idx]:
                    continue
                else:
                    complete_edge.append(merged_edge[idx])

            es = EdgeSegment()
            es.key_points = complete_edge
            es.pattern_idx = argmax_o  # This dasta segment is not uesd
            sum_val = 0
            max_val = 0
            for e in es.key_points:
                # sum_val += V[e[0], e[1]]
                if V[e[0], e[1]] > max_val:
                    max_val = V[e[0], e[1]]
            # es.val = sum_val / len(es.key_points)
            es.val = max_val
            ES.append(es)
    return ES


def run_model(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_img = img / img.max()

    V1_edge = Retina_V1(input_img)

    edge_nms = thin_edge(V1_edge)
    V2_LPs = LP_line(num_line=12, length=11, k_thresh=0.20, lr_thresh=0.10)
    V2SC_response, V2SC_l_response, V2SC_r_response = V2(edge_nms, V2_LPs)
    CS = EdgeSegmentDetectionAlgorithm(V2SC_response, V2SC_l_response, V2SC_r_response, V2_LPs)
    link_edge, edge_segment_coloring, edge_and_img = draw_continuous_edge(CS, input_img)

    edge = np.maximum(V1_edge, 1.2 * link_edge)
    edge /= edge.max()
    return edge, edge_segment_coloring


if __name__ == "__main__":

    cur_dir = os.path.abspath(os.curdir)
    img_path = os.path.join(cur_dir, "8068.jpg")
    edge, edge_segment_coloring = run_model(img_path)
    visualize_png(edge, "result_8068", cur_dir)
    visualize_png(edge_segment_coloring, "result_coloring_8068", cur_dir)