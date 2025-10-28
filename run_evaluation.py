
import gurobipy as gp
from gurobipy import GRB
import math 
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config
from exact_solver import solve_exact_decomposition
from baselines import solve_greedy_baseline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback
import os
os.environ['LC_ALL'] = 'C'  
os.environ['LANG'] = 'C'    
def create_problem_instance(N, seed=42, base_SA=500, p_mu=0.9, p_sigma=0.05):
    """
    Tạo một bộ tham số cho bài toán dựa trên số lượng node N.
    Sử dụng phân phối chuẩn cho p_i, C_i, B_i theo plan_test_v3.pdf.
    """
    print(f"Tạo dữ liệu cho N={N}, seed={seed}, base_SA={base_SA}, p_mu={p_mu}...")
    np.random.seed(seed) 
    random.seed(seed) 

    nodes = list(range(N))

    C_mu, C_sigma = 1250, 300
    C_raw = np.random.normal(C_mu, C_sigma, N)
    C = {i: max(1.0, C_raw[i]) for i in nodes} 

    B_mu, B_sigma = 0.050, 0.020
    B_raw = np.random.normal(B_mu, B_sigma, N)
    B = {i: max(0.001, B_raw[i]) for i in nodes}

    p_raw = np.random.normal(p_mu, p_sigma, N)
 
    p = {i: max(0.001, min(0.999, p_raw[i])) for i in nodes}

    params = {
        "S_A": base_SA,       
        "S_H": config.S_H,    
        "N_hot_min": config.N_hot_min,
        "tau_H": config.tau_H,
        "tau_A": config.tau_A,
        "gamma": config.gamma,
        "T_max": config.T_max, 
        "M": config.M,
        "N_nodes": N,
        "nodes": nodes,
        "C": C, "p": p, "B": B
    }

    if params["S_A"] * params["gamma"] > params["T_max"]:
        print(f"Cảnh báo: Bài toán có thể không khả thi (Decoding {params['S_A'] * params['gamma']:.2f}s > T_max {params['T_max']}s)")

    return params

def solve_monolithic(params, time_limit=1000):
    """
    Solves the original non-convex Storage Cost MINLP (from Final_paper_v2) using Gurobi.
    """
    solver_name = "Monolithic_Solver (Gurobi - Storage Cost)"
    print(f"Bắt đầu {solver_name}...")
    start_time = time.time()

    try:
        print("  [Debug] Đang tạo Gurobi Model...")
        env_params = {"OutputFlag": 0}
        with gp.Env(params=env_params) as env, gp.Model(name="Monolithic_Storage_MINLP", env=env) as m:
            print("  [Debug] Đã tạo Model. Đang trích xuất tham số...")

            N_nodes = params["N_nodes"]
            nodes = params["nodes"]
            S_A = params["S_A"]
            S_H = params["S_H"]
            C = params["C"]
            p = params["p"]
            B = params["B"]
            N_hot_min = params["N_hot_min"]
            tau_H = params["tau_H"]
            tau_A = params["tau_A"]
            gamma = params["gamma"]
            T_max = params["T_max"]
            M = params["M"]
            log_tau_H = math.log(tau_H) if tau_H > 0 else -math.inf
            log_tau_A = math.log(tau_A) if tau_A > 0 else -math.inf

            if log_tau_H == -math.inf or log_tau_A == -math.inf:
                 print(f"LỖI: tau_H ({tau_H}) hoặc tau_A ({tau_A}) không hợp lệ (<=0).")
                 return {"Algorithm": solver_name, "Cost_Z": None, "Time_s": 0, "n": None, "k": None, "Status": "Parameter Error"}
            print("  [Debug] Đã trích xuất tham số. Đang định nghĩa biến...")

            n = m.addVar(lb=1, ub=N_nodes, name="n", vtype=GRB.INTEGER)
            k = m.addVar(lb=1, name="k", vtype=GRB.INTEGER)
            x = m.addVars(nodes, name="x", vtype=GRB.BINARY)
            z = m.addVars(nodes, name="z", vtype=GRB.BINARY)
            y = m.addVars(nodes, name="y", vtype=GRB.BINARY)
            T_download = m.addVar(lb=0.0, name="T_download", vtype=GRB.CONTINUOUS)
            inv_k = m.addVar(lb=0.0, ub=1.0, name="inv_k")
            x_inv_k = m.addVars(nodes, lb=0.0, ub=1.0, name="x_inv_k")
            n_inv_k = m.addVar(lb=0.0, name="n_inv_k")
            print("  [Debug] Đã định nghĩa biến. Đang thêm ràng buộc...")

            m.addQConstr(k * inv_k == 1, name="inv_k_relation")
            m.addConstrs((x_inv_k[i] == x[i] * inv_k for i in nodes), name="bilinear_x_inv_k")
            m.addConstr(k <= n - 1, name="k_lt_n")
            m.addConstr(gp.quicksum(x[i] for i in nodes) == n, name="n_selection")
            m.addConstrs((x[i] + z[i] <= 1 for i in nodes), name="node_role_exclusive")
            m.addConstrs((z[i] * S_H + S_A * x_inv_k[i] <= C[i] for i in nodes), name="capacity")

            sum_zp = gp.quicksum(z[i] * p[i] for i in nodes)
            sum_z = gp.quicksum(z[i] for i in nodes)
            term_hot = sum_zp - N_hot_min + 1
            m.addQConstr(2 * term_hot * term_hot >= -log_tau_H * sum_z, name="hot_reliability")

            sum_xp = gp.quicksum(x[i] * p[i] for i in nodes)
            term_archive = sum_xp - k + 1
            m.addQConstr(2 * term_archive * term_archive >= -log_tau_A * n, name="archive_reliability")

            m.addConstr(gp.quicksum(y[i] for i in nodes) == k, name="k_finishers")
            m.addConstrs((y[i] <= x[i] for i in nodes), name="finisher_is_archive")

            for i in nodes:
                if B[i] > 0:
                     m.addConstr(T_download >= (S_A / B[i]) * inv_k - M * (1 - y[i]), name=f"T_download_lower_{i}")
                else:
                     m.addConstr(y[i] == 0, name=f"no_bw_finisher_{i}")

            m.addConstr(T_download + S_A * gamma <= T_max, name="total_time_limit")
            m.addConstr(n_inv_k == n * inv_k, name="bilinear_n_inv_k") 
            print("  [Debug] Đã thêm ràng buộc. Đang đặt hàm mục tiêu...") 
            if N_hot_min > 0: 
                m.addConstr(sum_z >= N_hot_min, name="min_hot_nodes_required")
            objective_expr = S_A * n_inv_k + S_H * sum_z
            m.setObjective(objective_expr, GRB.MINIMIZE)
            print("  [Debug] Đã đặt hàm mục tiêu. Đang đặt tham số Gurobi...") 

            m.Params.NonConvex = 2
            m.Params.TimeLimit = time_limit
            print(f"  [Debug] Đã đặt tham số (NonConvex=2, TimeLimit={time_limit}). Chuẩn bị gọi optimize...") 
            m.optimize()
            print("  [Debug] Đã gọi optimize xong. Đang xử lý kết quả...")

            status = m.Status
            elapsed_time = time.time() - start_time
   

            result_dict = {
                "Algorithm": solver_name,
                "Cost_Z": None,
                "Time_s": min(elapsed_time, time_limit),
                "n": None,
                "k": None,
                "sum_z": None,
                "Status": "Unknown"
            }

            if status == GRB.OPTIMAL:
                result_dict["Status"] = "Optimal"
                result_dict["Cost_Z"] = m.ObjVal
                result_dict["n"] = int(round(n.X))
                result_dict["k"] = int(round(k.X))
                try:
                    result_dict["sum_z"] = int(round(sum(z[i].X for i in nodes)))
                except AttributeError:
                    print("  -> Lỗi khi lấy giá trị sum_z cho Optimal.")
                    result_dict["sum_z"] = None 
                print(f"  -> {solver_name} found Optimal solution: Z = {m.ObjVal:.2f} in {elapsed_time:.2f}s (n={result_dict['n']}, k={result_dict['k']}, sz={result_dict['sum_z']})")
            elif status == GRB.TIME_LIMIT:
                result_dict["Status"] = "Timeout"
                print(f"  -> {solver_name} TIMEOUT after {time_limit}s.")
                if m.SolCount > 0:
                    result_dict["Cost_Z"] = m.ObjVal 
                    try:
                        result_dict["n"] = int(round(n.X))
                        result_dict["k"] = int(round(k.X))
                        result_dict["sum_z"] = int(round(sum(z[i].X for i in nodes)))
                    except AttributeError:
                        result_dict["n"] = None
                        result_dict["k"] = None
                        result_dict["sum_z"] = None
                        print("  -> Could not retrieve integer variable values before timeout.")
                    print(f"  -> Best feasible solution before timeout: Z = {m.ObjVal:.2f} (n={result_dict['n']}, k={result_dict['k']}, sz={result_dict['sum_z']})")
                else:
                     print(f"  -> No feasible solution found before timeout.")
            elif status == GRB.INFEASIBLE:
                result_dict["Status"] = "Infeasible"
                print(f"  -> {solver_name} determined the problem is Infeasible in {elapsed_time:.2f}s.")
            elif status == GRB.UNBOUNDED:
                 result_dict["Status"] = "Unbounded" 
                 print(f"  -> {solver_name} determined the problem is Unbounded in {elapsed_time:.2f}s.")
            else:
                result_dict["Status"] = f"Solver Status {status}"
                print(f"  -> {solver_name} finished with status code {status} in {elapsed_time:.2f}s.")
                if m.SolCount > 0: 
                     result_dict["Cost_Z"] = m.ObjVal
                     try:
                         result_dict["n"] = int(round(n.X))
                         result_dict["k"] = int(round(k.X))
                     except AttributeError:
                         result_dict["n"] = None
                         result_dict["k"] = None

            return result_dict

    except gp.GurobiError as e:
        print(f"Gurobi error occurred: {e.errno} ({e})")
        traceback.print_exc()
        return {
            "Algorithm": solver_name, "Cost_Z": None, "Time_s": time.time() - start_time,
            "n": None, "k": None, "Status": f"Gurobi Error {e.errno}"
         }
    except Exception as e:
        print(f"An unexpected error occurred in Monolithic Solver: {e}")
        traceback.print_exc()
        return {
             "Algorithm": solver_name, "Cost_Z": None, "Time_s": time.time() - start_time,
             "n": None, "k": None, "Status": f"Python Error: {type(e).__name__}"
         }

def run_experiment():
    """
    Chạy toàn bộ kịch bản đánh giá theo plan_test_v3.pdf.
    """
    print("--- Bắt đầu hàm run_experiment() ---")
    all_results = []
    solver_time_limit = 1000 
    print("\n===== BẮT ĐẦU KỊCH BẢN 1: SCALABILITY vs. N =====")
    N_values_scen1 = [20, 30, 40, 50, 60, 70, 80, 90, 100] 
    SA_scen1 = 500 
    p_mu_scen1 = 0.9 

    for N in N_values_scen1:
        print(f"\n--- Chạy Kịch bản 1 với N = {N} ---")
        params = create_problem_instance(N, seed=42, base_SA=SA_scen1, p_mu=p_mu_scen1)
        print(f"  [Debug] Chuẩn bị gọi solve_exact_decomposition cho N={N}...")

        result_hgida = solve_exact_decomposition(params)
        if result_hgida:
            all_results.append({
                "Scenario": 1, "N": N, "S_A": SA_scen1, "p_mu": p_mu_scen1,
                "Algorithm": "EIDA (Proposed)",
                "Cost_Z": result_hgida["Z_best"],
                "Time_s": result_hgida["computation_time"],
                "n": result_hgida["n"], "k": result_hgida["k"], "sum_z": result_hgida["sum_z"], "Status": "Optimal"
            })

        result_mono = solve_monolithic(params, time_limit=solver_time_limit)
        if result_mono:
             all_results.append({
                "Scenario": 1, "N": N, "S_A": SA_scen1, "p_mu": p_mu_scen1,
                **result_mono
             })

        result_greedy = solve_greedy_baseline(params)
        if result_greedy:
            all_results.append({
                "Scenario": 1, "N": N, "S_A": SA_scen1, "p_mu": p_mu_scen1,
                "Algorithm": "Greedy Baseline",
                "Cost_Z": result_greedy["Z_best"],
                "Time_s": result_greedy["computation_time"],
                "n": result_greedy["n"], "k": result_greedy["k"], "sum_z": result_greedy["sum_z"], "Status": "Optimal"
            })
        elif result_mono and result_mono["Status"] != "Timeout":
             all_results.append({
                "Scenario": 1, "N": N, "S_A": SA_scen1, "p_mu": p_mu_scen1,
                "Algorithm": "Greedy Baseline", "Cost_Z": None, "Time_s": None, "n": None, "k": None, "Status": "Infeasible/Failed"
             })

    print("\n===== BẮT ĐẦU KỊCH BẢN 2: SENSITIVITY vs. SA & RELIABILITY =====")
    N_scen2 = 50 
    SA_values_scen2 = [200, 500, 1000, 2000] 
    mu_values_scen2 = [0.8, 0.9, 0.95] 

    for SA_test in SA_values_scen2:
        for p_mu_test in mu_values_scen2:
            print(f"\n--- Chạy Kịch bản 2 với N={N_scen2}, SA={SA_test}, p_mu={p_mu_test} ---")
            params = create_problem_instance(N_scen2, seed=42, base_SA=SA_test, p_mu=p_mu_test)

            result_hgida = solve_exact_decomposition(params)
            if result_hgida:
                all_results.append({
                    "Scenario": 2, "N": N_scen2, "S_A": SA_test, "p_mu": p_mu_test,
                    "Algorithm": "EIDA (Proposed)",
                    "Cost_Z": result_hgida["Z_best"],
                    "Time_s": result_hgida["computation_time"],
                    "n": result_hgida["n"], "k": result_hgida["k"], "sum_z": result_hgida["sum_z"], "Status": "Optimal"
                })

            result_mono = solve_monolithic(params, time_limit=solver_time_limit)
            if result_mono:
                all_results.append({
                    "Scenario": 2, "N": N_scen2, "S_A": SA_test, "p_mu": p_mu_test,
                   **result_mono
                })

            result_greedy = solve_greedy_baseline(params)
            if result_greedy:
                 all_results.append({
                    "Scenario": 2, "N": N_scen2, "S_A": SA_test, "p_mu": p_mu_test,
                    "Algorithm": "Greedy Baseline",
                    "Cost_Z": result_greedy["Z_best"],
                    "Time_s": result_greedy["computation_time"],
                    "n": result_greedy["n"], "k": result_greedy["k"], "sum_z": result_greedy["sum_z"], "Status": "Optimal"
                })
            elif result_mono and result_mono["Status"] != "Timeout":
                 all_results.append({
                    "Scenario": 2, "N": N_scen2, "S_A": SA_test, "p_mu": p_mu_test,
                    "Algorithm": "Greedy Baseline", "Cost_Z": None, "Time_s": None, "n": None, "k": None, "Status": "Infeasible/Failed"
                 })

    if not all_results:
        print("Không có kết quả nào để phân tích.")
        return

    df = pd.DataFrame(all_results)
    df.sort_values(by=["Scenario", "N", "S_A", "p_mu", "Algorithm"], inplace=True) 
    print("\n--- KẾT QUẢ TỔNG HỢP ---")
    print(df.to_string()) 

    df.to_csv("evaluation_results_v3.csv", index=False)
    print("Đã lưu kết quả vào evaluation_results_v3.csv")

    plot_results(df)

def plot_results(df):
    """
    Vẽ biểu đồ từ DataFrame kết quả theo từng kịch bản.
    """
    print("Đang vẽ biểu đồ...")

    df_scen1 = df[df["Scenario"] == 1].copy() 
    if not df_scen1.empty:
        algorithms_scen1 = df_scen1["Algorithm"].unique()
        N_vals_scen1 = df_scen1["N"].unique()

        plt.figure(figsize=(10, 6))
        for algo in algorithms_scen1:
            df_plot = df_scen1[df_scen1["Algorithm"] == algo].dropna(subset=['Cost_Z']) 
            if not df_plot.empty:
                plt.plot(df_plot["N"], df_plot["Cost_Z"], marker='o', linestyle='-', label=f"Cost ({algo})")
        plt.xlabel("Số lượng Nodes (N)")
        plt.ylabel("Tổng chi phí (Z_best)")
        plt.xticks(N_vals_scen1) 
        plt.legend()
        plt.grid(True)
        plt.title("Kịch bản 1: So sánh Chi phí vs. Quy mô Mạng (N)")
        plt.savefig("plot_scen1_cost.png")
        print("Đã lưu biểu đồ plot_scen1_cost.png")
        plt.close() 
        plt.figure(figsize=(10, 6))
        for algo in algorithms_scen1:
            df_plot = df_scen1[df_scen1["Algorithm"] == algo].dropna(subset=['Time_s'])
            if not df_plot.empty:
                timeouts = df_plot[df_plot["Status"] == "Timeout"]
                non_timeouts = df_plot[df_plot["Status"] != "Timeout"]

                linestyle = '--' if algo == "Greedy Baseline" else '-' 
                marker = 'x' if algo.startswith("Monolithic") else 'o'

                if not non_timeouts.empty:
                     plt.plot(non_timeouts["N"], non_timeouts["Time_s"], marker=marker, linestyle=linestyle, label=f"Time ({algo})")
                if not timeouts.empty:

                     plt.scatter(timeouts["N"], timeouts["Time_s"], marker='x', color='red', s=100, zorder=5, label=f"Timeout ({algo})")

        plt.xlabel("Số lượng Nodes (N)")
        plt.ylabel("Thời gian tính toán (giây) - Log Scale")
        plt.xticks(N_vals_scen1)
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.title("Kịch bản 1: So sánh Thời gian chạy vs. Quy mô Mạng (N)")
        plt.savefig("plot_scen1_time.png")
        print("Đã lưu biểu đồ plot_scen1_time.png")
        plt.close()

    else:
        print("Không có dữ liệu cho Kịch bản 1 để vẽ đồ thị.")

    df_scen2 = df[df["Scenario"] == 2].copy()
    if not df_scen2.empty:
        print("\n--- Dữ liệu cho Kịch bản 2 (Sensitivity) ---")
        df_hgida_scen2 = df_scen2[df_scen2["Algorithm"] == "EIDA (Proposed)"]

        if not df_hgida_scen2.empty:
            print("Gợi ý vẽ biểu đồ 3D cho Kịch bản 2 (EIDA Cost vs SA vs p_mu):")
            print("Sử dụng thư viện như matplotlib.pyplot (Axes3D) hoặc plotly.")
            print("Trục X: S_A (Archive Data Size)")
            print("Trục Y: p_mu (Mean Availability)")
            print("Trục Z: Cost_Z (Total Cost)")
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            x = df_hgida_scen2['S_A']
            y = df_hgida_scen2['p_mu']
            z = df_hgida_scen2['Cost_Z']
            
            scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
            
            ax.set_xlabel('Archive Data Size (S_A, GB)')
            ax.set_ylabel('Mean Availability (p_mu)')
            ax.set_zlabel('Total Cost (Cost_Z)')
            ax.set_title('Kịch bản 2: EIDA Cost vs. S_A and Reliability')
            fig.colorbar(scatter, label='Cost_Z')
            plt.savefig("plot_scen2_3D_cost.png")
            print("Đã lưu (ví dụ) biểu đồ plot_scen2_3D_cost.png")
            plt.close()
            print("Đang vẽ biểu đồ 3D Stem...")
            fig_stem = plt.figure(figsize=(10, 8))
            ax_stem = fig_stem.add_subplot(111, projection='3d')

            for (xi, yi, zi) in zip(x, y, z):
                ax_stem.plot([xi, xi], [yi, yi], [0, zi], marker="_", color="grey", alpha=0.7)
            ax_stem.scatter(x, y, z, c=z, cmap='viridis', marker='o', depthshade=True)

            ax_stem.set_xlabel('Archive Data Size (S_A, GB)')
            ax_stem.set_ylabel('Mean Availability (p_mu)')
            ax_stem.set_zlabel('Total Cost (Cost_Z)')
            ax_stem.set_title('Kịch bản 2: EIDA Cost vs. S_A and Reliability (Stem Plot)')
            
            plt.savefig("plot_scen2_3D_stem.png") 
            print("Đã lưu biểu đồ plot_scen2_3D_stem.png")
            plt.close(fig_stem)
            print("Đang vẽ biểu đồ 3D Scatter cho sum_z...")
            fig_sumz = plt.figure(figsize=(10, 8))
            ax_sumz = fig_sumz.add_subplot(111, projection='3d')
            z_sumz = df_hgida_scen2['sum_z']
            scatter_sumz = ax_sumz.scatter(x, y, z_sumz, c=z_sumz, cmap='plasma', marker='o')
    
            ax_sumz.set_xlabel('Archive Data Size (S_A, GB)')
            ax_sumz.set_ylabel('Mean Availability (p_mu)')
            ax_sumz.set_zlabel('Số lượng Hot Replicas (sum_z)') 
            ax_sumz.set_title('Kịch bản 2: EIDA sum_z vs. S_A and Reliability') 
            
            fig_sumz.colorbar(scatter_sumz, label='sum_z') 
            
            plt.savefig("plot_scen2_3D_sum_z.png")
            print("Đã lưu biểu đồ plot_scen2_3D_sum_z.png")
            plt.close(fig_sumz)

            print("\nHoặc vẽ nhiều biểu đồ 2D:")
            print("- Cost vs SA (vẽ 3 đường cho 3 mức p_mu)")
            print("- Cost vs p_mu (vẽ nhiều đường cho các mức SA)")

        else:
             print("Không có dữ liệu EIDA cho Kịch bản 2.")

    else:
         print("Không có dữ liệu cho Kịch bản 2 để vẽ đồ thị.")


    print("\nHoàn thành vẽ và lưu biểu đồ.")
if __name__ == "__main__":
    print("--- Đang trong block __main__ ---")
    run_experiment()
