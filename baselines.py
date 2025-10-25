import time
import math
from exact_solver import check_feasibility, precalculate_bounds

def solve_greedy_baseline(params):
    """
    Một baseline tham lam:
    1. Cố định n = 70% tổng số node.
    2. Cố định k = n - 1 (để tối ưu chi phí S_A/k).
    3. Chỉ tìm sum_z nhỏ nhất (giống Vòng lặp 3 của thuật toán chính).
    """
    start_time = time.time()
    print("Bắt đầu thuật toán Greedy Baseline...")
    
    N = params["N_nodes"]
    params_copy = params.copy()
    params_copy = precalculate_bounds(params_copy) 

    n_test = math.floor(N * 0.7)
    if n_test < 2: n_test = 2
        
    k_test = n_test - 1
    
    if k_test <= 0:
        print("Không đủ node cho Greedy Baseline.")
        return None

    Z_best = float('inf')
    solution_best = None

    z_L = params_copy["N_hot_min"]
    z_U = N - n_test
    
    if z_L > z_U:
        print("Không có phạm vi z nào khả thi cho Greedy Baseline.")
        return None

    for z_test in range(z_L, z_U + 1):
        feasible, sub_solution = check_feasibility(n_test, k_test, z_test, params_copy)
        
        if feasible:
            Z_current = n_test * (params_copy["S_A"] / k_test) + z_test * params_copy["S_H"]
            
            if Z_current < Z_best:
                Z_best = Z_current
                solution_best = {
                    "n": n_test,
                    "k": k_test,
                    "sum_z": z_test,
                    "Z_best": Z_best,
                    "assignments": sub_solution
                }
                print(f"  -> Tìm thấy giải pháp Greedy: Z = {Z_best:.2f} (n={n_test}, k={k_test}, sz={z_test})")
            break 
            
    end_time = time.time()
    if solution_best:
        solution_best["computation_time"] = end_time - start_time
    
    if not solution_best:
        print("Greedy Baseline không tìm thấy giải pháp nào.")
        
    print(f"Hoàn thành Greedy Baseline. Thời gian chạy: {end_time - start_time:.2f}s")
    
    return solution_best