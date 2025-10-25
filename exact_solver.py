
import math
import time
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpBinary, PULP_CBC_CMD

def precalculate_bounds(params):
    """
    Tính toán trước các giá trị biên cần thiết
    """
    N = params["N_nodes"]
    nodes = params["nodes"]
    C_sorted = sorted(params["C"].items(), key=lambda item: item[1])
    params["C_n"] = {n+1: C_sorted[n][1] for n in range(N)}
    p_sorted = sorted(params["p"].items(), key=lambda item: item[1], reverse=True)
    p_vals_sorted = [val for (i, val) in p_sorted]
    params["P_n"] = {n+1: sum(p_vals_sorted[:n+1]) for n in range(N)}
    params["F_k_size"] = {}
    T_decode = params["S_A"] * params["gamma"]
    T_download_max = params["T_max"] - T_decode
    
    if T_download_max <= 0: 
        params["F_k_size"] = {k: 0 for k in range(1, N + 1)}
        return params

    for k in range(1, N + 1):
        min_B = params["S_A"] / (k * T_download_max) 
        F_k_set = {i for i in nodes if params["B"][i] >= min_B}
        params["F_k_size"][k] = len(F_k_set)
        
    return params

def get_k_min_cap(n, params):
    """
    Tính k_min từ ràng buộc dung lượng (C4), sử dụng logic đã sửa.
    k >= S_A / C_(N-n+1)
    """
    N = params["N_nodes"]
    idx = N - n + 1
    C_N_minus_n_plus_1 = params["C_n"].get(idx)
    
    if C_N_minus_n_plus_1 is None or C_N_minus_n_plus_1 == 0:
        return params["N_nodes"] + 1
    return math.ceil(params["S_A"] / C_N_minus_n_plus_1)

def get_k_max_avail(n, params):
    if n not in params["P_n"]:
         return 0
    P_n = params["P_n"][n]
    s_n = math.sqrt(-math.log(params["tau_A"]) * n / 2.0)
    return math.floor(P_n - s_n + 1)

def get_n_min_avail(k, params):
    N = params["N_nodes"]
    for n_prime in range(k + 1, N + 1):
        P_n_prime = params["P_n"].get(n_prime, 0)
        s_n_prime = math.sqrt(-math.log(params["tau_A"]) * n_prime / 2.0) 
        if P_n_prime >= (k - 1) + s_n_prime:
            return n_prime
    return N + 1 

def check_feasibility(n_param, k_param, sz_param, params):
    """
    [cite_start]Giải bài toán MIP Feasibility Problem [cite: 178, 233]
    """
    nodes = params["nodes"]
    prob = LpProblem(f"FeasibilityCheck_n{n_param}_k{k_param}_sz{sz_param}", LpMinimize)
    x = LpVariable.dicts("x", nodes, 0, 1, LpBinary) 
    z = LpVariable.dicts("z", nodes, 0, 1, LpBinary)
    y = LpVariable.dicts("y", nodes, 0, 1, LpBinary) 
    T_download = LpVariable("T_download", lowBound=0) 

    prob += lpSum(x[i] for i in nodes) == n_param 
    prob += lpSum(z[i] for i in nodes) == sz_param
    for i in nodes:
        prob += x[i] + z[i] <= 1
    frag_size = params["S_A"] / k_param
    for i in nodes:
        prob += z[i] * params["S_H"] + x[i] * frag_size <= params["C"][i]
    try:
        if params["tau_H"] <= 0 or params["tau_H"] >= 1:
            print(f"  ERROR: Invalid tau_H ({params['tau_H']})")
            return False, None
        if sz_param == 0:
            pass
        else:
            C_H_rhs = -math.log(params["tau_H"]) * sz_param
            sqrt_arg_h = C_H_rhs / 2.0
            if sqrt_arg_h < 0 and abs(sqrt_arg_h) < 1e-9:
                sqrt_arg_h = 0
            elif sqrt_arg_h < 0:
                print(f"  ERROR: Negative sqrt_arg_h ({sqrt_arg_h}) for Constant_H")
                return False, None

            Constant_H = (params["N_hot_min"] - 1) + math.sqrt(sqrt_arg_h)
            prob += lpSum(z[i] * params["p"][i] for i in nodes) >= Constant_H
    except ValueError as e:
        print(f"  ERROR calculating Constant_H (ValueError): {e}")
        return False, None
    except Exception as e:
        print(f"  UNEXPECTED ERROR calculating Constant_H: {e}")
        return False, None

    C_A_rhs = -math.log(params["tau_A"]) * n_param 
    Constant_A = (k_param - 1) + math.sqrt(C_A_rhs / 2.0) 
    prob += lpSum(x[i] * params["p"][i] for i in nodes) >= Constant_A

    prob += lpSum(y[i] for i in nodes) == k_param
    for i in nodes:
        prob += y[i] <= x[i]
        
    T_decode = params["S_A"] * params["gamma"]
    for i in nodes:
        if params["B"][i] > 0:
            time_i = (params["S_A"] / k_param) / params["B"][i]
            prob += T_download >= time_i - params["M"] * (1 - y[i])
        else:
            prob += y[i] == 0
    prob += T_download + T_decode <= params["T_max"]

    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    if LpStatus[prob.status] == "Optimal":
        solution = {
            "x": {i: int(x[i].varValue) for i in nodes},
            "z": {i: int(z[i].varValue) for i in nodes},
            "y": {i: int(y[i].varValue) for i in nodes},
            "T_download": T_download.varValue
        }
        return True, solution
    else:
        return False, None

def solve_exact_decomposition(params):

    start_time = time.time()
    params = precalculate_bounds(params)
    
    N = params["N_nodes"]
    Z_best = float('inf')
    solution_best = None 
    
    print("Bắt đầu thuật toán EIDA...")

    for n_test in range(2, N + 1):
        k_L = get_k_min_cap(n_test, params)
        k_U = get_k_max_avail(n_test, params)
        k_U = min(n_test - 1, k_U) 
        
        if k_L > k_U:
            continue
        for k_test in range(k_U, k_L - 1, -1):
            if params["F_k_size"].get(k_test, 0) < k_test:
                continue
            n_min = get_n_min_avail(k_test, params)
            if n_test < n_min:
                continue
            z_L = params["N_hot_min"]
            z_U = N - n_test         
            for z_test in range(z_L, z_U + 1):
                feasible, sub_solution = check_feasibility(n_test, k_test, z_test, params)
    
                if feasible: 
                    Z_current = n_test * (params["S_A"] / k_test) + z_test * params["S_H"] 
                    
                    if Z_current < Z_best:
                        Z_best = Z_current
                        solution_best = {
                            "n": n_test,
                            "k": k_test,
                            "sum_z": z_test,
                            "Z_best": Z_best,
                            "assignments": sub_solution
                        }
                        print(f"  -> Tìm thấy giải pháp mới: Z = {Z_best:.2f} (n={n_test}, k={k_test}, sz={z_test})")
                    break 

    end_time = time.time()
    if solution_best:
        solution_best["computation_time"] = end_time - start_time
    print(f"Hoàn thành EIDA. Thời gian chạy: {end_time - start_time:.2f}s")
    
    return solution_best
