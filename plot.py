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

    # --- Lọc dữ liệu cho Kịch bản 1 ---
    df_scen1 = df[df["Scenario"] == 1].copy()

    if df_scen1.empty:
        print("Không có dữ liệu cho Kịch bản 1 để vẽ đồ thị.")
        return

    # --- Chuẩn bị vẽ ---
    plt.figure(figsize=(10, 6))
    print("Đang xử lý dữ liệu để vẽ biểu đồ thời gian...")

    # Chỉ lấy thuật toán cần vẽ (Monolithic và EIDA)
    algorithms_to_plot = [
        "Monolithic_Solver (Gurobi - Storage Cost)",
        "EIDA (Proposed)"
    ]
    # Lấy giá trị N đã test
    N_vals_scen1 = sorted(df_scen1["N"].unique())

    # --- Vẽ từng đường cho mỗi thuật toán ---
    # Dùng list để lưu handles và labels cho legend sau này
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

        # ============================================
        # <<< BẮT ĐẦU TÙY CHỈNH THEO HÌNH ẢNH >>>
        # ============================================
        color = 'grey'
        marker = '.'
        linestyle = '-'
        label_time = "" # Sẽ đặt chính xác bên dưới
        label_timeout = "" # Sẽ đặt chính xác bên dưới

        if "Monolithic" in algo:
            color = 'blue'
            marker = 'x'
            label_time = "Time (Monolithic_Solver (Gurobi - Storage Cost))"
            label_timeout = "Timeout (Monolithic_Solver (Gurobi - Storage Cost))"
        elif "EIDA" in algo:
            color = 'darkorange'
            marker = 'o'
            label_time = "Time (EIDA (Proposed))"
            # EIDA không có timeout nên không cần label_timeout
        # ============================================
        # <<< KẾT THÚC TÙY CHỈNH >>>
        # ============================================

        # Vẽ đường non-timeout và lưu handle/label
        if not non_timeouts.empty:
            non_timeouts.sort_values(by='N', inplace=True)
            # plt.plot trả về một list các Line2D objects, lấy phần tử đầu tiên
            line, = plt.plot(non_timeouts["N"], non_timeouts["Time_s"],
                             color=color,
                             marker=marker,
                             linestyle=linestyle,
                             label=label_time) # Đặt label trực tiếp ở đây
            handles_list.append(line)
            labels_list.append(label_time)
            print(f"  Đã vẽ đường non-timeout cho '{algo}'")

        # Vẽ điểm timeout và lưu handle/label (chỉ cho Monolithic)
        if not timeouts.empty and "Monolithic" in algo:
            # plt.scatter trả về PathCollection object
            scatter = plt.scatter(timeouts["N"], timeouts["Time_s"],
                                marker='x', color='red', s=100, zorder=5,
                                label=label_timeout) # Đặt label trực tiếp ở đây
            handles_list.append(scatter)
            labels_list.append(label_timeout)
            print(f"  Đã vẽ điểm timeout cho '{algo}'")

    # --- Cấu hình các thành phần của biểu đồ ---
    plt.xlabel("Number of Nodes (N)")
    plt.ylabel("Execution Time (seconds)")
    # plt.yscale('log')
    plt.xticks(N_vals_scen1)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.title("Runtime Comparison vs. Network Size (N)")

    # --- Tạo Legend với thứ tự chính xác ---
    # Thứ tự mong muốn: Time Mono, Timeout Mono, Time EIDA
    desired_order = [
        "Time (Monolithic_Solver (Gurobi - Storage Cost))",
        "Timeout (Monolithic_Solver (Gurobi - Storage Cost))",
        "Time (EIDA (Proposed))"
    ]

    # Sắp xếp lại handles và labels theo desired_order
    ordered_handles = []
    ordered_labels = []
    # Tạo một dict để dễ tìm index
    label_to_handle = dict(zip(labels_list, handles_list))

    for label in desired_order:
        if label in label_to_handle:
            ordered_handles.append(label_to_handle[label])
            ordered_labels.append(label)

    # Chỉ hiển thị legend nếu có mục nào đó được vẽ
    if ordered_handles:
        plt.legend(ordered_handles, ordered_labels, loc='best')
    else:
        print("Cảnh báo: Không có dữ liệu để hiển thị legend.")


    # --- Lưu biểu đồ ---
    output_filename = "plot_scen1_time.pdf" # Đổi tên file output
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

    # --- Lọc dữ liệu cho Kịch bản 1 ---
    df_scen1 = df[df["Scenario"] == 1].copy()

    if df_scen1.empty:
        print("Không có dữ liệu cho Kịch bản 1 để vẽ đồ thị.")
        return

    # --- Chuẩn bị vẽ ---
    plt.figure(figsize=(10, 6))
    print("Đang xử lý dữ liệu để vẽ biểu đồ chi phí...")

    # Chỉ lấy thuật toán cần vẽ (Monolithic và EIDA)
    algorithms_to_plot = [
        "Monolithic_Solver (Gurobi - Storage Cost)",
        "EIDA (Proposed)"
        # Bỏ qua Greedy nếu không muốn vẽ
    ]
    N_vals_scen1 = sorted(df_scen1["N"].unique())

    # Dùng list để lưu handles và labels cho legend
    handles_list = []
    labels_list = []

    # --- Vẽ từng đường cho mỗi thuật toán ---
    for algo in algorithms_to_plot:
        df_algo = df_scen1[df_scen1["Algorithm"] == algo].copy()
        df_algo['Cost_Z'] = pd.to_numeric(df_algo['Cost_Z'], errors='coerce')
        df_algo.dropna(subset=['Cost_Z'], inplace=True)
        if df_algo.empty:
            print(f" Bỏ qua thuật toán '{algo}' vì thiếu dữ liệu Cost_Z hợp lệ.")
            continue

        # ============================================
        # <<< BẮT ĐẦU TÙY CHỈNH THEO HÌNH ẢNH >>>
        # ============================================
        color = 'grey'
        marker = '.'
        linestyle = '-'
        label = "" # Sẽ đặt chính xác bên dưới

        if "Monolithic" in algo:
            color = 'blue'       # Màu xanh dương
            marker = 'o'         # Marker 'o'
            label = "Cost (Monolithic_Solver (Gurobi - Storage Cost))" # Label chính xác
        elif "EIDA" in algo:
            color = 'darkorange' # Màu cam đậm
            marker = 'o'         # Marker 'o'
            label = "Cost (EIDA (Proposed))" # Label chính xác
        # ============================================
        # <<< KẾT THÚC TÙY CHỈNH >>>
        # ============================================

        # Vẽ đường và lưu handle/label
        df_algo.sort_values(by='N', inplace=True)
        line, = plt.plot(df_algo["N"], df_algo["Cost_Z"],
                         color=color,
                         marker=marker,
                         linestyle=linestyle,
                         label=label) # Đặt label trực tiếp
        handles_list.append(line)
        labels_list.append(label)
        print(f"  Đã vẽ đường chi phí cho '{algo}' ({len(df_algo)} điểm)")

    # --- Cấu hình các thành phần của biểu đồ ---
    plt.xlabel("Number of Nodes (N)")
    plt.ylabel("Total Cost (Z_best)")
    plt.xticks(N_vals_scen1)
    plt.grid(True)
    plt.title("Cost vs. Network Size (N)")

    # --- Tạo Legend với thứ tự chính xác ---
    # Thứ tự mong muốn: Monolithic trước, EIDA sau
    desired_order = [
        "Cost (Monolithic_Solver (Gurobi - Storage Cost))",
        "Cost (EIDA (Proposed))"
    ]

    # Sắp xếp lại handles và labels
    ordered_handles = []
    ordered_labels = []
    label_to_handle = dict(zip(labels_list, handles_list))

    for label in desired_order:
        if label in label_to_handle:
            ordered_handles.append(label_to_handle[label])
            ordered_labels.append(label)

    # Chỉ hiển thị legend nếu có mục nào đó được vẽ
    if ordered_handles:
        plt.legend(ordered_handles, ordered_labels, loc='best')
    else:
        print("Cảnh báo: Không có dữ liệu để hiển thị legend.")


    # --- Lưu biểu đồ ra file PDF ---
    output_filename = "plot_scen1_cost.pdf" # Đuôi .pdf
    output_filepath = os.path.join(script_dir, output_filename)
    try:
        plt.savefig(output_filepath, format='pdf') # Lưu dạng pdf
        print(f"\nĐã lưu biểu đồ vào {output_filepath}")
    except Exception as e:
        print(f"\nLỗi khi lưu biểu đồ: {e}")

    plt.close()
# --- Gọi hàm chính để thực thi ---
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

    # --- Lọc dữ liệu cho Kịch bản 2 và thuật toán EIDA ---
    df_scen2 = df[df["Scenario"] == 2].copy()
    df_eida_scen2 = df_scen2[df_scen2["Algorithm"] == "EIDA (Proposed)"].copy()

    # Bỏ qua nếu không có dữ liệu EIDA hợp lệ
    df_eida_scen2['Cost_Z'] = pd.to_numeric(df_eida_scen2['Cost_Z'], errors='coerce')
    df_eida_scen2.dropna(subset=['S_A', 'p_mu', 'Cost_Z'], inplace=True)

    if df_eida_scen2.empty:
        print("Không có dữ liệu EIDA hợp lệ cho Kịch bản 2 để vẽ biểu đồ 3D Cost.")
        return

    # --- Chuẩn bị vẽ 3D ---
    print("Đang xử lý dữ liệu để vẽ biểu đồ 3D Cost...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Lấy dữ liệu cho các trục
    x = df_eida_scen2['S_A']
    y = df_eida_scen2['p_mu']
    z = df_eida_scen2['Cost_Z']

    # --- Vẽ 3D Scatter Plot ---
    # c=z: màu sắc điểm dựa trên giá trị z (Cost_Z)
    # cmap='viridis': Bảng màu giống hình ảnh
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=50) # s=50: kích thước điểm

    # --- Cấu hình các thành phần của biểu đồ ---
    ax.set_xlabel('Archive Data Size (S_A, GB)')
    ax.set_ylabel('Mean Availability (p_mu)')
    ax.set_zlabel('Total Cost (Cost_Z)')
    ax.set_title('EIDA Cost vs. S_A and Reliability')

    # Thêm colorbar để giải thích màu sắc
    fig.colorbar(scatter, label='Cost_Z', shrink=0.7) # shrink: làm colorbar ngắn lại chút

    # --- Lưu biểu đồ ra file PDF ---
    output_filename = "plot_scen2_3D_cost.pdf" # <<< Đuôi .pdf
    output_filepath = os.path.join(script_dir, output_filename)
    try:
        plt.savefig(output_filepath, format='pdf', bbox_inches='tight') # <<< Lưu PDF, bbox_inches='tight' để không bị cắt rìa
        print(f"\nĐã lưu biểu đồ vào {output_filepath}")
    except Exception as e:
        print(f"\nLỗi khi lưu biểu đồ 3D Cost: {e}")

    plt.close(fig) # Đóng figure này

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

    # --- Lọc dữ liệu cho Kịch bản 2 và thuật toán EIDA ---
    df_scen2 = df[df["Scenario"] == 2].copy()
    # Giữ tên biến df_hgida_scen2 cho nhất quán với code cũ, dù chứa data EIDA
    df_hgida_scen2 = df_scen2[df_scen2["Algorithm"] == "EIDA (Proposed)"].copy()

    # Bỏ qua nếu không có dữ liệu EIDA hợp lệ
    df_hgida_scen2['Cost_Z'] = pd.to_numeric(df_hgida_scen2['Cost_Z'], errors='coerce')
    df_hgida_scen2.dropna(subset=['S_A', 'p_mu', 'Cost_Z'], inplace=True)

    if df_hgida_scen2.empty:
        print("Không có dữ liệu EIDA hợp lệ cho Kịch bản 2 để vẽ biểu đồ 3D Stem Cost.")
        return

    # --- Chuẩn bị vẽ 3D ---
    print("Đang xử lý dữ liệu để vẽ biểu đồ 3D Stem Cost...")
    fig_stem = plt.figure(figsize=(10, 8))
    ax_stem = fig_stem.add_subplot(111, projection='3d')

    # Lấy dữ liệu cho các trục
    x = df_hgida_scen2['S_A']
    y = df_hgida_scen2['p_mu']
    z = df_hgida_scen2['Cost_Z']

    # --- Vẽ 3D Stem Plot thủ công ---
    for (xi, yi, zi) in zip(x, y, z):
        # Vẽ đường thẳng đứng (stem) từ (xi, yi, 0) đến (xi, yi, zi)
        # marker='_' ở điểm gốc (z=0) để làm rõ chân stem
        ax_stem.plot([xi, xi], [yi, yi], [0, zi], marker="_", markersize=10, color="grey", alpha=0.7, zorder=1) # zorder=1 để vẽ sau marker

    # Vẽ các điểm marker ở trên cùng (sau các stem)
    # Dùng scatter để tô màu theo giá trị z
    scatter_stem = ax_stem.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=50, depthshade=True, zorder=2) # zorder=2 để vẽ trên stem

    # --- Cấu hình các thành phần của biểu đồ ---
    ax_stem.set_xlabel('Archive Data Size (S_A, GB)')
    ax_stem.set_ylabel('Mean Availability (p_mu)')
    ax_stem.set_zlabel('Total Cost (Cost_Z)')
    ax_stem.set_title('EIDA Cost vs. S_A and Reliability (Stem Plot)')

    # Thêm colorbar (giống scatter plot)
    fig_stem.colorbar(scatter_stem, label='Cost_Z', shrink=0.7)

    # --- Lưu biểu đồ ra file PDF ---
    output_filename = "plot_scen2_3D_stem.pdf" # <<< Đuôi .pdf
    output_filepath = os.path.join(script_dir, output_filename)
    try:
        plt.savefig(output_filepath, format='pdf', bbox_inches='tight') # <<< Lưu PDF
        print(f"\nĐã lưu biểu đồ vào {output_filepath}")
    except Exception as e:
        print(f"\nLỗi khi lưu biểu đồ 3D Stem Cost: {e}")

    plt.close(fig_stem) # Đóng figure này

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

    # --- Lọc dữ liệu cho Kịch bản 2 và thuật toán EIDA ---
    df_scen2 = df[df["Scenario"] == 2].copy()
    # Giữ tên biến df_hgida_scen2 cho nhất quán, dù chứa data EIDA
    df_hgida_scen2 = df_scen2[df_scen2["Algorithm"] == "EIDA (Proposed)"].copy()

    # Bỏ qua nếu không có dữ liệu EIDA hợp lệ cho các cột cần thiết
    # Chuyển đổi sum_z sang số, lỗi sẽ thành NaN
    df_hgida_scen2['sum_z'] = pd.to_numeric(df_hgida_scen2['sum_z'], errors='coerce')
    df_hgida_scen2.dropna(subset=['S_A', 'p_mu', 'sum_z'], inplace=True) # <-- Cần cột sum_z

    if df_hgida_scen2.empty:
        print("Không có dữ liệu EIDA hợp lệ cho Kịch bản 2 để vẽ biểu đồ 3D sum_z.")
        return

    # --- Chuẩn bị vẽ 3D ---
    print("Đang xử lý dữ liệu để vẽ biểu đồ 3D sum_z...")
    fig_sumz = plt.figure(figsize=(10, 8))
    ax_sumz = fig_sumz.add_subplot(111, projection='3d')

    # Lấy dữ liệu cho các trục
    x = df_hgida_scen2['S_A']
    y = df_hgida_scen2['p_mu']
    z_sumz = df_hgida_scen2['sum_z'] # <-- Trục Z là sum_z

    # --- Vẽ 3D Scatter Plot ---
    # c=z_sumz: màu sắc điểm dựa trên giá trị z (sum_z)
    # cmap='plasma': Bảng màu thường dùng cho giá trị nguyên hoặc phân tán
    scatter_sumz = ax_sumz.scatter(x, y, z_sumz, c=z_sumz, cmap='plasma', marker='o', s=50)

    # --- Cấu hình các thành phần của biểu đồ ---
    ax_sumz.set_xlabel('Archive Data Size (S_A, GB)')
    ax_sumz.set_ylabel('Mean Availability (p_mu)')
    ax_sumz.set_zlabel('Number of Hot Replicas (sum_z)') # <-- Label trục Z
    ax_sumz.set_title('EIDA sum_z vs. S_A and Reliability') # <-- Tiêu đề

    # Thêm colorbar để giải thích màu sắc
    fig_sumz.colorbar(scatter_sumz, label='sum_z', shrink=0.7) # <-- Label colorbar

    # --- Lưu biểu đồ ra file PDF ---
    output_filename = "plot_scen2_3D_sum_z.pdf" # <<< Đuôi .pdf
    output_filepath = os.path.join(script_dir, output_filename)
    try:
        plt.savefig(output_filepath, format='pdf', bbox_inches='tight') # <<< Lưu PDF
        print(f"\nĐã lưu biểu đồ vào {output_filepath}")
    except Exception as e:
        print(f"\nLỗi khi lưu biểu đồ 3D sum_z: {e}")

    plt.close(fig_sumz) # Đóng figure này

if __name__ == "__main__":
    # Đảm bảo bạn đang chạy file này từ thư mục chứa file CSV
    # Hoặc cung cấp đường dẫn đầy đủ đến file CSV
    plot_scenario1_time()
    plot_scenario1_cost()
    plot_scenario2_cost_3d()
    plot_scenario2_cost_3d_stem()
    plot_scenario2_sumz_3d()
    print("Hoàn thành.")