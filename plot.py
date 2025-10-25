import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_scenario1_time(csv_filename="evaluation_results_v3.csv"):
    """
    Vẽ biểu đồ Thời gian chạy vs. Quy mô Mạng (N) cho Kịch bản 1
    từ file CSV kết quả. Tùy chỉnh màu sắc, marker, và thứ tự legend
    để khớp chính xác với hình ảnh mẫu.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filepath = os.path.join(script_dir, csv_filename)

    try:
        df = pd.read_csv(csv_filepath)
        print(f"Đã đọc dữ liệu từ {csv_filepath}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_filepath}")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return
    df_scen1 = df[df["Scenario"] == 1].copy()

    if df_scen1.empty:
        print("Không có dữ liệu cho Kịch bản 1 để vẽ đồ thị.")
        return
    
    plt.figure(figsize=(10, 6))
    print("Đang xử lý dữ liệu để vẽ biểu đồ thời gian...")

    algorithms_to_plot = [
        "Monolithic_Solver (Gurobi - Storage Cost)",
        "EIDA (Proposed)"
    ]
    N_vals_scen1 = sorted(df_scen1["N"].unique())

    handles_list = []
    labels_list = []

    for algo in algorithms_to_plot:
        df_algo = df_scen1[df_scen1["Algorithm"] == algo].copy()
        df_algo['Time_s'] = pd.to_numeric(df_algo['Time_s'], errors='coerce')
        df_algo.dropna(subset=['Time_s'], inplace=True)
        if df_algo.empty:
            print(f" Bỏ qua thuật toán '{algo}' vì thiếu dữ liệu Time_s hợp lệ.")
            continue

        timeouts = df_algo[df_algo["Status"] == "Timeout"].copy()
        non_timeouts = df_algo[df_algo["Status"] != "Timeout"].copy()

        color = 'grey'
        marker = '.'
        linestyle = '-'
        label_time = "" 
        label_timeout = "" 

        if "Monolithic" in algo:
            color = 'blue'
            marker = 'x'
            label_time = "Time (Commercial_Solver)"
            label_timeout = "Timeout (Commercial_Solver)"
        elif "EIDA" in algo:
            color = 'darkorange'
            marker = 'o'
            label_time = "Time (GDA (Proposed))"

        if not non_timeouts.empty:
            non_timeouts.sort_values(by='N', inplace=True)

            line, = plt.plot(non_timeouts["N"], non_timeouts["Time_s"],
                             color=color,
                             marker=marker,
                             linestyle=linestyle,
                             label=label_time)
            handles_list.append(line)
            labels_list.append(label_time)
            print(f"  Đã vẽ đường non-timeout cho '{algo}'")

        if not timeouts.empty and "Monolithic" in algo:

            scatter = plt.scatter(timeouts["N"], timeouts["Time_s"],
                                marker='x', color='red', s=100, zorder=5,
                                label=label_timeout)
            handles_list.append(scatter)
            labels_list.append(label_timeout)
            print(f"  Đã vẽ điểm timeout cho '{algo}'")

    fontsize_medium = 18 
    fontsize_legend = 12

    plt.xlabel("Number of Nodes (N)", fontsize=fontsize_medium)
    plt.ylabel("Execution Time (seconds)", fontsize=fontsize_medium)
    plt.xticks(N_vals_scen1, fontsize=fontsize_medium)
    plt.yticks(fontsize=fontsize_medium)
    plt.grid(True, which="both", ls="-", alpha=0.5)

    desired_order = [
        "Time (Commercial_Solver)",
        "Timeout (Commercial_Solver)",
        "Time (GDA (Proposed))"
    ]

    ordered_handles = []
    ordered_labels = []
    label_to_handle = dict(zip(labels_list, handles_list))

    for label in desired_order:
        if label in label_to_handle:
            ordered_handles.append(label_to_handle[label])
            ordered_labels.append(label)

    if ordered_handles:
        plt.legend(ordered_handles, ordered_labels, loc='best', fontsize=fontsize_legend)
    else:
        print("Cảnh báo: Không có dữ liệu để hiển thị legend.")

    output_filename = "plot_scen1_time.pdf" 
    output_filepath = os.path.join(script_dir, output_filename)
    try:
        plt.savefig(output_filepath, format='pdf')
        print(f"\nĐã lưu biểu đồ vào {output_filepath}")
    except Exception as e:
        print(f"\nLỗi khi lưu biểu đồ: {e}")

    plt.close()
def plot_scenario1_cost(csv_filename="evaluation_results_v3.csv"):
    """
    Vẽ biểu đồ So sánh Chi phí vs. Quy mô Mạng (N) cho Kịch bản 1
    từ file CSV kết quả, lưu PDF, và khớp legend với hình ảnh mẫu.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filepath = os.path.join(script_dir, csv_filename)

    try:
        df = pd.read_csv(csv_filepath)
        print(f"Đã đọc dữ liệu từ {csv_filepath}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_filepath}")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return

    df_scen1 = df[df["Scenario"] == 1].copy()

    if df_scen1.empty:
        print("Không có dữ liệu cho Kịch bản 1 để vẽ đồ thị.")
        return

    plt.figure(figsize=(10, 6))
    print("Đang xử lý dữ liệu để vẽ biểu đồ chi phí...")

    algorithms_to_plot = [
        "Monolithic_Solver (Gurobi - Storage Cost)",
        "EIDA (Proposed)"
    ]
    N_vals_scen1 = sorted(df_scen1["N"].unique())

    handles_list = []
    labels_list = []

    for algo in algorithms_to_plot:
        df_algo = df_scen1[df_scen1["Algorithm"] == algo].copy()
        df_algo['Cost_Z'] = pd.to_numeric(df_algo['Cost_Z'], errors='coerce')
        df_algo.dropna(subset=['Cost_Z'], inplace=True)
        if df_algo.empty:
            print(f" Bỏ qua thuật toán '{algo}' vì thiếu dữ liệu Cost_Z hợp lệ.")
            continue

        color = 'grey'
        marker = '.'
        linestyle = '-'
        label = ""

        if "Monolithic" in algo:
            color = 'blue'      
            marker = 'o'        
            label = "Cost (Commercial_Solver)" 
        elif "EIDA" in algo:
            color = 'darkorange' 
            marker = 'o'         
            label = "Cost (GDA (Proposed))" 
        df_algo.sort_values(by='N', inplace=True)
        line, = plt.plot(df_algo["N"], df_algo["Cost_Z"],
                         color=color,
                         marker=marker,
                         linestyle=linestyle,
                         label=label) 
        handles_list.append(line)
        labels_list.append(label)
        print(f"  Đã vẽ đường chi phí cho '{algo}' ({len(df_algo)} điểm)")

    fontsize_medium = 18
    fontsize_legend = 12

    plt.xlabel("Number of Nodes (N)", fontsize=fontsize_medium)
    plt.ylabel("Total Cost (Z_best)", fontsize=fontsize_medium)
    plt.xticks(N_vals_scen1, fontsize=fontsize_medium)
    plt.yticks(fontsize=fontsize_medium)
    plt.grid(True)
    desired_order = [
        "Cost (Commercial_Solver)",
        "Cost (GDA (Proposed))"
    ]

    ordered_handles = []
    ordered_labels = []
    label_to_handle = dict(zip(labels_list, handles_list))

    for label in desired_order:
        if label in label_to_handle:
            ordered_handles.append(label_to_handle[label])
            ordered_labels.append(label)

    if ordered_handles:
        plt.legend(ordered_handles, ordered_labels, loc='best', fontsize=fontsize_legend)
    else:
        print("Cảnh báo: Không có dữ liệu để hiển thị legend.")

    output_filename = "plot_scen1_cost.pdf" 
    output_filepath = os.path.join(script_dir, output_filename)
    try:
        plt.savefig(output_filepath, format='pdf') 
        print(f"\nĐã lưu biểu đồ vào {output_filepath}")
    except Exception as e:
        print(f"\nLỗi khi lưu biểu đồ: {e}")

    plt.close()
def plot_scenario2_cost_3d(csv_filename="evaluation_results_v3.csv"):
    """
    Vẽ biểu đồ 3D Scatter Plot (EIDA Cost_Z vs. S_A vs. p_mu) cho Kịch bản 2
    từ file CSV kết quả và lưu dưới dạng PDF.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filepath = os.path.join(script_dir, csv_filename)

    try:
        df = pd.read_csv(csv_filepath)
        print(f"Đã đọc dữ liệu từ {csv_filepath} cho biểu đồ 3D Cost.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_filepath}")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return

    df_scen2 = df[df["Scenario"] == 2].copy()
    df_eida_scen2 = df_scen2[df_scen2["Algorithm"] == "EIDA (Proposed)"].copy()

    df_eida_scen2['Cost_Z'] = pd.to_numeric(df_eida_scen2['Cost_Z'], errors='coerce')
    df_eida_scen2.dropna(subset=['S_A', 'p_mu', 'Cost_Z'], inplace=True)

    if df_eida_scen2.empty:
        print("Không có dữ liệu EIDA hợp lệ cho Kịch bản 2 để vẽ biểu đồ 3D Cost.")
        return

    print("Đang xử lý dữ liệu để vẽ biểu đồ 3D Cost...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = df_eida_scen2['S_A']
    y = df_eida_scen2['p_mu']
    z = df_eida_scen2['Cost_Z']

    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=50) 

    ax.set_xlabel('Archive Data Size (S_A, GB)')
    ax.set_ylabel('Mean Availability (p_mu)')
    ax.set_zlabel('Total Cost (Cost_Z)')
    ax.set_title('EIDA Cost vs. S_A and Reliability')

    fig.colorbar(scatter, label='Cost_Z', shrink=0.7)

    output_filename = "plot_scen2_3D_cost.pdf" 
    output_filepath = os.path.join(script_dir, output_filename)
    try:
        plt.savefig(output_filepath, format='pdf', bbox_inches='tight')
        print(f"\nĐã lưu biểu đồ vào {output_filepath}")
    except Exception as e:
        print(f"\nLỗi khi lưu biểu đồ 3D Cost: {e}")

    plt.close(fig) 

def plot_scenario2_cost_3d_stem(csv_filename="evaluation_results_v3.csv"):
    """
    Vẽ biểu đồ 3D Stem Plot (EIDA Cost_Z vs. S_A vs. p_mu) cho Kịch bản 2
    từ file CSV kết quả và lưu dưới dạng PDF.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filepath = os.path.join(script_dir, csv_filename)

    try:
        df = pd.read_csv(csv_filepath)
        print(f"Đã đọc dữ liệu từ {csv_filepath} cho biểu đồ 3D Stem Cost.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_filepath}")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return

    df_scen2 = df[df["Scenario"] == 2].copy()
    df_hgida_scen2 = df_scen2[df_scen2["Algorithm"] == "EIDA (Proposed)"].copy()

    df_hgida_scen2['Cost_Z'] = pd.to_numeric(df_hgida_scen2['Cost_Z'], errors='coerce')
    df_hgida_scen2.dropna(subset=['S_A', 'p_mu', 'Cost_Z'], inplace=True)

    if df_hgida_scen2.empty:
        print("Không có dữ liệu EIDA hợp lệ cho Kịch bản 2 để vẽ biểu đồ 3D Stem Cost.")
        return

    print("Đang xử lý dữ liệu để vẽ biểu đồ 3D Stem Cost...")
    fig_stem = plt.figure(figsize=(10, 8))
    ax_stem = fig_stem.add_subplot(111, projection='3d')

    x = df_hgida_scen2['S_A']
    y = df_hgida_scen2['p_mu']
    z = df_hgida_scen2['Cost_Z']

    for (xi, yi, zi) in zip(x, y, z):
        ax_stem.plot([xi, xi], [yi, yi], [0, zi], marker="_", markersize=10, color="grey", alpha=0.7, zorder=1) 

    scatter_stem = ax_stem.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=50, depthshade=True, zorder=2) 

    fontsize_medium = 12

    ax_stem.set_xlabel('Archive Data Size (S_A, GB)', fontsize=fontsize_medium)
    ax_stem.set_ylabel('Mean Availability (p_mu)', fontsize=fontsize_medium)
    ax_stem.set_zlabel('Total Cost (Cost_Z)', fontsize=fontsize_medium)
    ax_stem.tick_params(axis='x', labelsize=fontsize_medium) 
    ax_stem.tick_params(axis='y', labelsize=fontsize_medium) 
    ax_stem.tick_params(axis='z', labelsize=fontsize_medium) 

    cbar = fig_stem.colorbar(scatter_stem, label='Cost_Z', shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize_medium) 
    cbar.set_label('Cost_Z', size=fontsize_medium) 

    output_filename = "plot_scen2_3D_stem.pdf" 
    output_filepath = os.path.join(script_dir, output_filename)
    try:
        plt.savefig(output_filepath, format='pdf', bbox_inches='tight') 
        print(f"\nĐã lưu biểu đồ vào {output_filepath}")
    except Exception as e:
        print(f"\nLỗi khi lưu biểu đồ 3D Stem Cost: {e}")

    plt.close(fig_stem) 

def plot_scenario2_sumz_3d(csv_filename="evaluation_results_v3.csv"):
    """
    Vẽ biểu đồ 3D Scatter Plot (EIDA sum_z vs. S_A vs. p_mu) cho Kịch bản 2
    từ file CSV kết quả và lưu dưới dạng PDF.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filepath = os.path.join(script_dir, csv_filename)

    try:
        df = pd.read_csv(csv_filepath)
        print(f"Đã đọc dữ liệu từ {csv_filepath} cho biểu đồ 3D sum_z.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_filepath}")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return

    df_scen2 = df[df["Scenario"] == 2].copy()
    df_hgida_scen2 = df_scen2[df_scen2["Algorithm"] == "EIDA (Proposed)"].copy()
    df_hgida_scen2['sum_z'] = pd.to_numeric(df_hgida_scen2['sum_z'], errors='coerce')
    df_hgida_scen2.dropna(subset=['S_A', 'p_mu', 'sum_z'], inplace=True) 

    if df_hgida_scen2.empty:
        print("Không có dữ liệu EIDA hợp lệ cho Kịch bản 2 để vẽ biểu đồ 3D sum_z.")
        return

    print("Đang xử lý dữ liệu để vẽ biểu đồ 3D sum_z...")
    fig_sumz = plt.figure(figsize=(10, 8))
    ax_sumz = fig_sumz.add_subplot(111, projection='3d')

    x = df_hgida_scen2['S_A']
    y = df_hgida_scen2['p_mu']
    z_sumz = df_hgida_scen2['sum_z'] 
    scatter_sumz = ax_sumz.scatter(x, y, z_sumz, c=z_sumz, cmap='plasma', marker='o', s=50)

    ax_sumz.set_xlabel('Archive Data Size (S_A, GB)')
    ax_sumz.set_ylabel('Mean Availability (p_mu)')
    ax_sumz.set_zlabel('Number of Hot Replicas (sum_z)') 
    ax_sumz.set_title('EIDA sum_z vs. S_A and Reliability') 

    fig_sumz.colorbar(scatter_sumz, label='sum_z', shrink=0.7) 

    output_filename = "plot_scen2_3D_sum_z.pdf" 
    output_filepath = os.path.join(script_dir, output_filename)
    try:
        plt.savefig(output_filepath, format='pdf', bbox_inches='tight') 
        print(f"\nĐã lưu biểu đồ vào {output_filepath}")
    except Exception as e:
        print(f"\nLỗi khi lưu biểu đồ 3D sum_z: {e}")

    plt.close(fig_sumz) 

if __name__ == "__main__":
    plot_scenario1_time()
    plot_scenario1_cost()
    plot_scenario2_cost_3d()
    plot_scenario2_cost_3d_stem()
    plot_scenario2_sumz_3d()
    print("Hoàn thành.")