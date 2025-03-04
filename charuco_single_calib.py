import scipy.optimize
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import scipy
import time
from argparse import ArgumentParser


# cam1 = glob("./calib_0120/Cam_001/*.png")
# cam1 = glob("C:/Users/kimdo/OneDrive - UNIST/2024 Winter/calibration/new_cam/*")
cam1 = glob("C:/Users/kimdo/Desktop/calib_data_filtered (1)/calib_data/Cam_001/*.png")
cam2 = glob("C:/Users/kimdo/Desktop/calib_data_filtered (1)/calib_data/Cam_002/*.png")

board_size = (5, 5)
length_square = 0.098  # 사각형 크기 (미터 단위)
length_marker = 0.073  # 마커 크기 (미터 단위)

id_to_coord = {0: (0, 0, 1), 1: (0, 1, 1), 2: (0, 2, 1), 3: (0, 3, 1), 4: (1, 0, 1), 5: (1, 1, 1), 6: (1, 2, 1), 7: (1, 3, 1), 8: (2, 0, 1), 9: (2, 1, 1), 10: (2, 2, 1), 11: (2, 3, 1), 12: (3, 0, 1),
               13: (3, 1, 1), 14: (3, 2, 1), 15: (3, 3, 1)}

def v_ij(i, j, H):
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[1, i] * H[2, j] + H[2, i] * H[1, j],
        H[2, i] * H[2, j]
    ])

def reprojection_error(params, imgpoints, objpoints):
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = params[:9]
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    # print(f"K: {K}")
    residuals = []
    extrinsics = params[9:].reshape(-1, 3)
    num_img = int(len(extrinsics) / 2)
    
    for i, (X_s, X_w) in enumerate(zip(imgpoints, objpoints)):
        r_i = extrinsics[i]
        t_i = extrinsics[i + num_img]
        R_i, _ = cv2.Rodrigues(r_i)
        # print(f"r_i: {r_i}")
        # print(f"t_i: {t_i}")
        for X_w_j, X_s_j in zip(X_w, X_s):
            X_c = R_i @ X_w_j + t_i
            x, y = X_c[0] / X_c[2], X_c[1] / X_c[2]
            r_sqr = x * x + y * y
            rad_dist = (1 + k1 * r_sqr + k2 * r_sqr * r_sqr + k3 * r_sqr * r_sqr * r_sqr)
            tan_dist = np.array([
                2 * p1 * x * y + p2 * (r_sqr + 2 * x * x),
                2 * p2 * x * y + p1 * (r_sqr + 2 * y * y)
            ])
            X_distorted =  rad_dist * np.array([x, y]) + tan_dist
            X_hat = np.dot(K, np.append(X_distorted, 1))
            residuals.append(X_s_j[:2] - X_hat[:2])
    print(np.sqrt(np.sum(np.linalg.norm(np.array(residuals), axis=1) ** 2) / len(residuals)))      # np.linalg.norm default: matrix - frobenius norm, vector - l2-norm
    return np.array(residuals).flatten()

def main():
    parser = ArgumentParser()
    parser.add_argument('cam', help='Camera number (1: left or 2: right)')
    args = parser.parse_args()
    if args.cam == 'cam1':
        cam = cam1
    else:
        cam = cam2

    V = np.empty((2 * len(cam), 6))
    V_cnt = 0
    H_list = []
    objpoints = []
    imgpoints = []
    
    for path in tqdm(cam):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (1400, 1050), interpolation=cv2.INTER_LINEAR)
        # 체크보드 코너 찾기
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard(board_size, length_square, length_marker, dictionary)
        c_detector = cv2.aruco.CharucoDetector(board)
        corners, ids, _, _ = c_detector.detectBoard(img)
        # corners: (16, 1, 2)
        corners = np.squeeze(corners, 1)        # corners: (16, 2)
        new_col = [[1]] * corners.shape[0]
        corners = np.c_[corners, new_col]       # corners: (16, 3), last column = 1
        
        imgpoints.append(corners)
        
        A = np.empty((2 * corners.shape[0], 9))     # (2n, 9) (n: # of corners)
        
        X_wj = np.zeros((corners.shape[0], 3))

        for i in range(corners.shape[0]):
            X_s = corners[i]                # [pixel]
            X_w = np.array(id_to_coord[ids[i][0]])   # [meter]

            X_wj[i, :] = X_w
            col0 = np.concatenate((X_w, np.zeros(3), np.array([-X_s[0] * i for i in X_w])))
            col1 = np.concatenate((np.zeros(3), X_w, np.array([-X_s[1] * i for i in X_w])))
            A[i * 2, :] = col0
            A[i * 2 + 1, :] = col1
            
        objpoints.append(X_wj)
        # Singular Vector Decomposition for DLT (Direct Linear Transformation)
        U, S, Vh = np.linalg.svd(A)
        # find right singular vector with smallest singular value
        h = Vh[-1]
        H = h.reshape(3, 3)
        H /= H[2, 2]        # normalize H
        # print(f"H = {H}")
        H_list.append(H)
        # =========================================================================================
        # # stacking to form V. (Vb=0 => find intrinsic params.)
        # v12 = np.array([
        #     H[0, 0]*H[1, 0],
        #     H[0, 0]*H[1, 1] + H[0, 1]*H[1, 0],
        #     H[0, 1]*H[1, 1],
        #     H[0, 2]*H[1, 0] + H[0, 0]*H[1, 2],
        #     H[0, 1]*H[1, 2] + H[0, 2]*H[1, 1],
        #     H[0, 2]*H[1, 2]
        # ])
        # v22 = np.array([
        #     H[0, 0]*H[0, 0] - H[1, 0]*H[1, 0],
        #     2 * H[0, 0]*H[0, 1] - 2 * H[1, 0]*H[1, 1],
        #     H[0, 1]*H[0, 1] - H[1, 1]*H[1, 1],
        #     2 * H[0, 2]*H[0, 0] - 2 * H[1, 2]*H[1, 0],
        #     2 * H[0, 1]*H[0, 2] - 2 * H[1, 1]*H[1, 2],
        #     H[0, 2]*H[0, 2] - H[1, 2]*H[1, 2]
        # ])

        # V[V_cnt, :] = v12
        # V[V_cnt+1, :] = v22
        # =========================================================================================

        V[V_cnt, :] = v_ij(0, 1, H)
        V[V_cnt+1, :] = v_ij(0, 0, H) - v_ij(1, 1, H)
        V_cnt += 2

    # print(f'V={V}')
    # print(f"# of detected images: {len(detected_imgs)}")

    # closed-form solution for camera instrinsic parameters
    U, D, Vt = np.linalg.svd(V)
    b = Vt[-1]      # b = [B11, B12, B22, B13, B23, B33]^T
    # print(f"b: {b}")
    # print(f'norm b: {np.linalg.norm(b)}')
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])  # B = lambda * K^-TK^-1 => find K using cholesky decomposition
    B /= B[2, 2]

    K_inv_T = np.linalg.cholesky(B)
    K_inv = K_inv_T.T
    K = np.linalg.inv(K_inv)
    K /= K[2, 2]

    # print(f'K = {K}')

    # cy = (B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2]) / (B[0, 0]*B[1, 1] - B[0, 1]*B[0, 1])
    # l = B[2, 2] - (B[0, 2]*B[0, 2] + cy * (B[0, 1]*B[0,2] - B[0, 0]*B[1, 2])) / B[0, 0]       # lambda: scaling factor
    # fx = np.sqrt(l / B[0, 0])
    # fy = np.sqrt(l * B[0, 0] / (B[0, 0]*B[1, 1] - B[0, 1]*B[0, 1]))
    # s = -B[0, 1] * fx * fx * fy / l
    # cx = s * cy / fy - B[0, 2] * fx * fx / l

    # K = np.array([
    #     [fx, s, cx],
    #     [0, fy, cy],
    #     [0, 0, 1]
    # ])


    R_list = []
    t_list = []
    # closed-form solution for camera extrinsic parameters
    for i in range(len(cam)):
        R_i = np.empty((3, 3))
        l = 1 / np.linalg.norm(np.matmul(np.linalg.inv(K), H_list[i][:, 0]))
        r1 = l * np.matmul(np.linalg.inv(K), H_list[i][:, 0])
        r2 = l * np.matmul(np.linalg.inv(K), H_list[i][:, 1])
        r3 = np.cross(r1, r2)
        t_list.append(l * np.matmul(np.linalg.inv(K), H_list[i][:, 2]))
        R_i[:, 0] = r1
        R_i[:, 1] = r2
        R_i[:, 2] = r3
        R_list.append(R_i)
        # print(f'r1: {r1}')
        # print(f"r1 norm: {np.linalg.norm(r1)}, r2 norm: {np.linalg.norm(r2)}, r3 norm: {np.linalg.norm(r3)}")

    # change rotation matrices to rotation vectors
    r_i = np.zeros((len(cam), 3))
    for i, R in enumerate(R_list):
        r_vec, _ = cv2.Rodrigues(R)
        r_i[i, :] = r_vec[:, 0]
    intrinsic_params = np.hstack([K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0, 0, 0, 0, 0])       # fx, fy, cx, cy, k1, k2, p1, p2, k3
    extrinsic_params = np.hstack([r.flatten() for r in r_i] + [t.flatten() for t in t_list])
    params = np.hstack([intrinsic_params, extrinsic_params])
    start = time.time()
    res = scipy.optimize.least_squares(reprojection_error, np.array(params), method='lm', args=[imgpoints, objpoints])
    end = time.time()
    print("The time of LM optimization algorithm is :",
            (end-start) / 60, "min")
    print('---------------------------')
    res_K = np.array([
        [res.x[0],  0,          res.x[2]],
        [0,         res.x[1],   res.x[3]],
        [0,         0,             1    ]
    ])
    res_dist = np.array([res.x[4], res.x[5], res.x[6], res.x[7], res.x[8]])
    print(f"Intrinsic Matrix: {res_K}")
    print(f"Distortion Coef: {res_dist}")
    # print(f'res.cost: {res.cost}')
    # print(f'res.optimality: {res.optimality}')
    print(f'res.x: {res.x}')
    res_extrinsics = res.x[9:]
    rvecs = res_extrinsics[:int(res_extrinsics.shape[0]/2)]
    tvecs = res_extrinsics[int(res_extrinsics.shape[0]/2):]
    print(f"rvecs: {rvecs}")
    print(f"tvecs: {tvecs}")
    return [res.x, imgpoints, objpoints]

if __name__ == '__main__':
    main()

