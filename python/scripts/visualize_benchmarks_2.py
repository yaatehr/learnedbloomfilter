import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3 as sql
import pandas as pd
from math import log10, floor


round_to_n = lambda x, n: round(x, -int(floor(log10(abs(x)))) + (n - 1))

# test_name = "explicit_always_false"
test_name = "explicit_lstm_1"
fixture_data = pd.read_csv(f"{test_name}_normalized.csv", index_col=False)

fixture_data["calibration_fpr"] = fixture_data.inv_max_fpr.astype(float)**-1
# columns = ["function", "projected_eles", "inv_max_fpr", "actual_fpr", "num_hashes", "table_size", "batch_size", "len_gen_eles", "iterations", "real_time", "cpu_time"]
columns = ["function", "actual_fpr", "num_hashes", "calibration_fpr", "table_size", "batch_size"]
fixture_data = fixture_data[columns]
# print(fixture_data)


# string_query = fixture_data[(fixture_data.function.str.contains("TestBloomFilterStringQuery"))]
# string_insertion = fixture_data[(fixture_data.function.str.contains("TestBloomFilterStringInsertion"))]
# int_insertion = fixture_data[(fixture_data.function.str.contains("TestBloomFilterIntInsertion"))]
# int_query = fixture_data[(fixture_data.function.str.contains("TestBloomFilterIntQuery"))]


# we want to gorup by batch_size because fprs are comperable
# fpr==num_hashes should be shown against the table size
# batch-size, calculated_fpr, then map table size against acutal fpr

queries = [fixture_data]
querie_name = ["string_query"]
# sort_by = ["projected_eles", "fpr", "pos_eles", "ele_length"]
sort_by = ["projected_eles", "fpr", "pos_eles"]
time_aggregate = ["real_time", "cpu_time"]
colors = ["b", "c", "g", "y", "r", "m", "k"]
unique_hashes = fixture_data.calibration_fpr.unique()
unique_hashes.sort()
# print(unique_hashes)
colormap = dict(zip(unique_hashes, colors[: len(unique_hashes)]))

for ind, query in enumerate(queries): 
    for group1_label, group1 in query.groupby("batch_size"):
        g = group1.sort_values("actual_fpr")
        # if(len(g.actual_fpr.unique()) == 1):
        #     continue

        fig, ax = plt.subplots()
        for group2_label, group2 in g.groupby("calibration_fpr"):
                group2_label_r = round_to_n(group2_label, 3)
                ax.plot(group2.actual_fpr, group2.table_size, marker='o', linestyle='', label=group2_label_r, color=colormap[group2_label])
        ax.legend()
        ax.title.set_text("%s, batch_size: %d" % (test_name, group1_label))
        ax.set_ylabel("table size (bytes)")
        plt.savefig(f"laptop_benchmarks/{test_name}/batch_size-{group1_label}.png")
        plt.clf()
        plt.close()
    for group1_label, group1 in query.groupby("batch_size"):
        g = group1.sort_values("actual_fpr")
        # if(len(g.actual_fpr.unique()) == 1):
        #     continue

        fig, ax = plt.subplots()
        for group2_label, group2 in g.groupby("calibration_fpr"):
                group2_label_r = round_to_n(group2_label, 3)

                ax.plot(group2.actual_fpr, group2.table_size, marker='o', linestyle='', label=group2_label_r, color=colormap[group2_label])
        ax.legend()
        ax.set_yscale("log")
        ax.title.set_text("%s, batch_size: %d" % (test_name, group1_label))
        ax.set_ylabel("table size (bytes)")
        plt.savefig(f"laptop_benchmarks/{test_name}/batch_size-{group1_label}-logy.png")
        plt.clf()
        plt.close()
    for group1_label, group1 in query.groupby("batch_size"):
        g = group1.sort_values("actual_fpr")
        # if(len(g.actual_fpr.unique()) == 1):
        #     continue

        for group2_label, group2 in g.groupby("calibration_fpr"):
                group2_label_r = round_to_n(group2_label, 3)

                fig, ax = plt.subplots()
                ax.plot(group2.actual_fpr, group2.table_size, marker='o', linestyle='', label=group2_label_r, color=colormap[group2_label])
                # ax.legend()
                ax.set_yscale("log")
                ax.title.set_text("%s, batch_size: %d" % (test_name, group1_label))
                ax.set_ylabel("table size (bytes)")
                plt.savefig(f"laptop_benchmarks/{test_name}/batch_size-{group1_label}-frp-{group2_label_r}-logy.png")
                plt.clf()
                plt.close()
