
import math
import time
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpBinary, PULP_CBC_CMD

# ===================================================================
# [cite_start]PHẦN 1: CÁC HÀM TÍNH TOÁN BIÊN (Bounds) [cite: 134-175]
# ===================================================================

def precalculate_bounds(params):
    """
    Tính toán trước các giá trị biên cần thiết
    """
    N = params["N_nodes"]
    nodes = params["nodes"]
    
    # [cite_start]Sắp xếp C (tăng dần) [cite: 152]
    C_sorted = sorted(params["C"].items(), key=lambda item: item[1])
    params["C_n"] = {n+1: C_sorted[n][1] for n in range(N)} # C_(n) [cite: 153]
    
    # [cite_start]Sắp xếp p (giảm dần) [cite: 165]
    p_sorted = sorted(params["p"].items(), key=lambda item: item[1], reverse=True)
    p_vals_sorted = [val for (i, val) in p_sorted]
    
    # [cite_start]P_n: Tổng tiền tố của p_sorted [cite: 165]
    params["P_n"] = {n+1: sum(p_vals_sorted[:n+1]) for n in range(N)}
    
    # [cite_start]F(k): Tập các node đủ nhanh [cite: 143-144]
    params["F_k_size"] = {}
    T_decode = params["S_A"] * params["gamma"] # [cite: 20-21, 98]
    T_download_max = params["T_max"] - T_decode # [cite: 139]
    
    if T_download_max <= 0: # [cite: 137]
        params["F_k_size"] = {k: 0 for k in range(1, N + 1)}
        return params

    for k in range(1, N + 1):
        # [cite: 144]
        min_B = params["S_A"] / (k * T_download_max) 
        F_k_set = {i for i in nodes if params["B"][i] >= min_B}
        params["F_k_size"][k] = len(F_k_set) # |F(k)| [cite: 147]
        
    return params

def get_k_min_cap(n, params):
    """
    Tính k_min từ ràng buộc dung lượng (C4), sử dụng logic đã sửa.
    k >= S_A / C_(N-n+1)
    """
    N = params["N_nodes"]
    
    # Logic cũ (SAI): C_n = params["C_n"].get(n)
    
    # Logic mới (ĐÚNG): Lấy node C_(N-n+1)
    # Đây là node có dung lượng nhỏ nhất trong số N-n+1 node lớn nhất
    # (hay còn gọi là node lớn thứ n)
    idx = N - n + 1
    C_N_minus_n_plus_1 = params["C_n"].get(idx)
    
    if C_N_minus_n_plus_1 is None or C_N_minus_n_plus_1 == 0:
        return params["N_nodes"] + 1 # Không khả thi
        
    # Sửa công thức [cite: 154-155]
    return math.ceil(params["S_A"] / C_N_minus_n_plus_1)

def get_k_max_avail(n, params):
    # [cite_start]k <= P_n - s(n) + 1 [cite: 168]
    if n not in params["P_n"]:
         return 0 # Không khả thi
    P_n = params["P_n"][n] # [cite: 165]
    s_n = math.sqrt(-math.log(params["tau_A"]) * n / 2.0) # s(n) [cite: 159]
    return math.floor(P_n - s_n + 1) # [cite: 168]

def get_n_min_avail(k, params):
    # n >= min{n' | [cite_start]P_n' >= (k-1) + s(n')} [cite: 170]
    N = params["N_nodes"]
    for n_prime in range(k + 1, N + 1): # n' phải >= k+1 [cite: 56]
        P_n_prime = params["P_n"].get(n_prime, 0)
        s_n_prime = math.sqrt(-math.log(params["tau_A"]) * n_prime / 2.0) # [cite: 159]
        if P_n_prime >= (k - 1) + s_n_prime: # [cite: 166, 170]
            return n_prime
    return N + 1 # Không tìm thấy n' khả thi

# ===================================================================
# [cite_start]PHẦN 2: LÕI MIP - CHECK FEASIBILITY [cite: 176]
# ===================================================================

def check_feasibility(n_param, k_param, sz_param, params):
    """
    [cite_start]Giải bài toán MIP Feasibility Problem [cite: 178, 233]
    """
    nodes = params["nodes"]
    
    # 1. Khởi tạo mô hình
    prob = LpProblem(f"FeasibilityCheck_n{n_param}_k{k_param}_sz{sz_param}", LpMinimize)

    # [cite_start]2. Định nghĩa Biến số [cite: 49, 50, 51, 52, 242]
    x = LpVariable.dicts("x", nodes, 0, 1, LpBinary) # x_i [cite: 49]
    z = LpVariable.dicts("z", nodes, 0, 1, LpBinary) # z_i [cite: 50]
    y = LpVariable.dicts("y", nodes, 0, 1, LpBinary) # y_i [cite: 51]
    T_download = LpVariable("T_download", lowBound=0) # T_download [cite: 52]
    
    # [cite_start]3. Thêm các ràng buộc (Constraints) [cite: 243-265]
    
    # Ràng buộc cố định từ các vòng lặp ngoài
    prob += lpSum(x[i] for i in nodes) == n_param  # sum(x_i) = n [cite: 57, 244]
    prob += lpSum(z[i] for i in nodes) == sz_param # sum(z_i) = sz [cite: 182, 244]
    
    # [cite_start]Ràng buộc Logic (C3) [cite: 58, 244]
    for i in nodes:
        prob += x[i] + z[i] <= 1
    
    # [cite_start]Ràng buộc Dung lượng (C4) - Đã tuyến tính hóa [cite: 183, 245]
    frag_size = params["S_A"] / k_param # [cite: 60]
    for i in nodes:
        prob += z[i] * params["S_H"] + x[i] * frag_size <= params["C"][i] # [cite: 60, 245]

    # [cite_start]Ràng buộc Tin cậy Hot (C5) - Đã tuyến tính hóa [cite: 184-186, 248]
    try:
        # Kiểm tra tau_H hợp lệ trước
        if params["tau_H"] <= 0 or params["tau_H"] >= 1:
            print(f"  ERROR: Invalid tau_H ({params['tau_H']})") # Thêm print lỗi
            return False, None
        if sz_param == 0:
            # Ràng buộc gốc (6) luôn đúng khi sum(z)=0 (vế trái >= 0, vế phải = 0)
            # Không cần thêm ràng buộc tuyến tính hóa (vì nó sẽ thành 0 >= N_hot_min - 1)
            pass # Bỏ qua, không thêm ràng buộc C5
        else:
            # Chỉ tính và thêm ràng buộc khi sz_param > 0
            C_H_rhs = -math.log(params["tau_H"]) * sz_param
            sqrt_arg_h = C_H_rhs / 2.0
            # Xử lý số âm rất nhỏ do làm tròn số (~0)
            if sqrt_arg_h < 0 and abs(sqrt_arg_h) < 1e-9:
                sqrt_arg_h = 0
            elif sqrt_arg_h < 0: # Số thực sự âm
                print(f"  ERROR: Negative sqrt_arg_h ({sqrt_arg_h}) for Constant_H")
                return False, None # Tham số căn bậc hai âm -> không khả thi

            Constant_H = (params["N_hot_min"] - 1) + math.sqrt(sqrt_arg_h)
            prob += lpSum(z[i] * params["p"][i] for i in nodes) >= Constant_H
    except ValueError as e: # Bắt lỗi miền giá trị từ log hoặc sqrt
        print(f"  ERROR calculating Constant_H (ValueError): {e}")
        return False, None
    except Exception as e: # Bắt các lỗi khác
        print(f"  UNEXPECTED ERROR calculating Constant_H: {e}")
        return False, None

    # [cite_start]Ràng buộc Tin cậy Archive (C6) - Đã tuyến tính hóa [cite: 187-188, 249]
    C_A_rhs = -math.log(params["tau_A"]) * n_param # [cite: 62, 188]
    Constant_A = (k_param - 1) + math.sqrt(C_A_rhs / 2.0) # [cite: 188]
    prob += lpSum(x[i] * params["p"][i] for i in nodes) >= Constant_A # [cite: 188, 247]

    # [cite_start]Ràng buộc Thời gian (C7) - Đã tuyến tính hóa [cite: 189]
    prob += lpSum(y[i] for i in nodes) == k_param # C7a [cite: 87, 256]
    for i in nodes:
        prob += y[i] <= x[i] # C7b [cite: 97, 256]
        
    T_decode = params["S_A"] * params["gamma"] # [cite: 20-21]
    for i in nodes:
        # [cite_start]C7c: Big-M [cite: 190, 264]
        if params["B"][i] > 0:
            time_i = (params["S_A"] / k_param) / params["B"][i] # [cite: 88, 139]
            prob += T_download >= time_i - params["M"] * (1 - y[i]) # [cite: 88, 264]
        else:
            prob += y[i] == 0 # Node B=0 không thể là fastest finisher

    # [cite_start]C7d: (Dựa trên ràng buộc 11 gốc) [cite: 98, 265]
    prob += T_download + T_decode <= params["T_max"]

    # [cite_start]4. Giải bài toán [cite: 262]
    solver = PULP_CBC_CMD(msg=False) # Ẩn output của solver
    prob.solve(solver)
    
    # [cite_start]5. Trả về kết quả [cite: 267]
    if LpStatus[prob.status] == "Optimal": # "Optimal" nghĩa là tìm thấy giải pháp khả thi
        solution = {
            "x": {i: int(x[i].varValue) for i in nodes},
            "z": {i: int(z[i].varValue) for i in nodes},
            "y": {i: int(y[i].varValue) for i in nodes},
            "T_download": T_download.varValue
        }
        return True, solution
    else:
        return False, None

# ===================================================================
# [cite_start]PHẦN 3: THUẬT TOÁN CHÍNH (Algorithm 1) [cite: 191]
# ===================================================================

def solve_exact_decomposition(params):

    start_time = time.time()
    
    # [cite_start]1. Tiền xử lý (Line 3) [cite: 194]
    params = precalculate_bounds(params)
    
    N = params["N_nodes"]
    Z_best = float('inf') # [cite: 195]
    solution_best = None # [cite: 195]
    
    print("Bắt đầu thuật toán EIDA...")

    # [cite_start]2. Vòng lặp 1: n (Line 4) [cite: 118, 196]
    for n_test in range(2, N + 1):
        
        # [cite_start]3. Tính toán biên k (Line 5) [cite: 197]
        k_L = get_k_min_cap(n_test, params) # [cite: 173]
        k_U = get_k_max_avail(n_test, params) # [cite: 173]
        k_U = min(n_test - 1, k_U) # k phải < n [cite: 56]
        
        if k_L > k_U:
            continue

        # [cite_start]4. Vòng lặp 2: k (Line 6) [cite: 121, 199]
        # [cite_start]Duyệt từ cao xuống thấp (directed search) [cite: 125-126]
        for k_test in range(k_U, k_L - 1, -1):
            
            # [cite_start]5. Kiểm tra biên (Line 7-9) [cite: 201-207]
            if params["F_k_size"].get(k_test, 0) < k_test: # |F(k)| >[cite_start]= k [cite: 147, 173]
                continue
            
            n_min = get_n_min_avail(k_test, params)
            if n_test < n_min: # [cite: 174]
                continue
                
            # [cite_start]6. Vòng lặp 3: sum(z_i) (Line 10) [cite: 127, 211]
            z_L = 0 # [cite: 129, 209]
            z_U = N - n_test          # [cite: 209]
            
            # [cite_start]Duyệt từ thấp lên cao (directed search) [cite: 128-129]
            for z_test in range(z_L, z_U + 1):
                
                # [cite_start]7. Gọi lõi MIP (Line 11) [cite: 216]
                feasible, sub_solution = check_feasibility(n_test, k_test, z_test, params)
                
                # [cite_start]8. Xử lý kết quả (Line 12-18) [cite: 218]
                if feasible: 
                    # [cite_start]Tính toán chi phí [cite: 220-223]
                    Z_current = n_test * (params["S_A"] / k_test) + z_test * params["S_H"] # [cite: 59]
                    
                    if Z_current < Z_best: # [cite: 225]
                        Z_best = Z_current # [cite: 227]
                        solution_best = { # [cite: 236]
                            "n": n_test,
                            "k": k_test,
                            "sum_z": z_test,
                            "Z_best": Z_best,
                            "assignments": sub_solution
                        }
                        print(f"  -> Tìm thấy giải pháp mới: Z = {Z_best:.2f} (n={n_test}, k={k_test}, sz={z_test})")
                        
                    # [cite_start]Thoát vòng lặp 3 (Line 18) [cite: 238]
                    # [cite_start]Vì đã tìm thấy sz nhỏ nhất cho (n, k) này [cite: 130]
                    break 

    end_time = time.time()
    if solution_best:
        solution_best["computation_time"] = end_time - start_time
    print(f"Hoàn thành EIDA. Thời gian chạy: {end_time - start_time:.2f}s")
    
    return solution_best
