import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3 as sql
import pandas as pd
from math import log10, floor
import os

round_to_n = lambda x, n: round(x, -int(floor(log10(abs(x)))) + (n - 1))

# test_name = "explicit_always_false"
# test_name = "timestamp_lstm_3"
test_name = "timestamp_embedding_lstm_4"
# test_name = "timestamp_gru_1"
fixture_data = pd.read_csv(f"../../input/{test_name}.csv", index_col=False)

fixture_data["compound_size"] = fixture_data["table_size"] + fixture_data["lbf_size"]
print(fixture_data.columns)
print(fixture_data["num_eles_tested"][0])

fixture_data["ns_per_insert"] = fixture_data["insert_time"]/fixture_data["num_eles_tested"][0] #TODO fix, these do not work
fixture_data["ns_per_query"] = fixture_data["query_time"]/fixture_data["num_eles_tested"][0] #TODO fix, these do not work

# columns = ["function", "projected_eles", "inv_max_fpr", "empirical_fpr", "num_hashes", "table_size", "batch_size", "len_gen_eles", "iterations", "real_time", "cpu_time"]
# columns = ["empirical_fpr", "num_hashes", "target_fpr", "tau", "table_size", "compound_size", "ns_per_insert", "ns_per_query", "num_eles_tested", "projected_fallback_count","projected_fallback_percentage", "fallback_count","gbf_effective_fpr"]
columns = ["empirical_fpr", "num_hashes", "target_fpr", "tau", "table_size", "compound_size", "ns_per_insert", "ns_per_query", "num_eles_tested", "projected_fallback_count","projected_fallback_percentage", "fallback_count","gbf_effective_fpr"]

fixture_data = fixture_data[columns]

print(fixture_data.head(10))


# string_query = fixture_data[(fixture_data.function.str.contains("TestBloomFilterStringQuery"))]
# string_insertion = fixture_data[(fixture_data.function.str.contains("TlbestBloomFilterStringInsertion"))]
# int_insertion = fixture_data[(fixture_data.function.str.contains("TestBloomFilterIntInsertion"))]
# int_query = fixture_data[(fixture_data.function.str.contains("TestBloomFilterIntQuery"))]

gbf_data = fixture_data[(fixture_data["tau"] ==1)]
gbf_data = gbf_data.sort_values("table_size")
lbf_data = fixture_data[(fixture_data["tau"] !=1)]
print(gbf_data.tail(10))


# we want to gorup by batch_size because fprs are comperable
# fpr==num_hashes should be shown against the table size
# batch-size, calculated_fpr, then map table size against acutal fpr

queries = [lbf_data]
querie_name = ["string_query"]
# sort_by = ["projected_eles", "fpr", "pos_eles", "ele_length"]
sort_by = ["projected_eles", "fpr", "pos_eles"]
time_aggregate = ["real_time", "cpu_time"]
colors = ["b", "c", "g", "y", "r", "m", "k"]
unique_tau = fixture_data.tau.unique()
# unique_hashes.sort()
# print(unique_hashes)
colormap = dict(zip(unique_tau, colors[: len(unique_tau)]))
if not os.path.exists(f"laptop_benchmarks/{test_name}"):
    os.makedirs(f"laptop_benchmarks/{test_name}")

for ind, query in enumerate(queries): 
    for group1_label, group1 in query.groupby("tau"):
        g = group1.sort_values("compound_size")
        print(g.tail(10))
        fig, ax = plt.subplots()
        ax.plot(g.compound_size, g.empirical_fpr, marker='.', linestyle='-', label=group1_label, color="b")
        ax.plot(gbf_data.table_size, gbf_data.empirical_fpr, marker='.', linestyle='-', label="Generic BF", color="r")
        ax.legend()
        ax.title.set_text("%s, tau: %d" % (test_name, group1_label))
        ax.set_ylabel("Empirical FPR")
        ax.set_xlabel("Table Size (bytes)")
        plt.savefig(f"laptop_benchmarks/{test_name}/tau-{group1_label}.png")
        plt.clf()
        plt.close()

    for group1_label, group1 in query.groupby("tau"):
        g = group1.sort_values("compound_size")
        fig, ax = plt.subplots()
        ax.plot(g.compound_size, g.empirical_fpr, marker='.', linestyle='-', label=group1_label, color="b")
        ax.plot(gbf_data.table_size, gbf_data.empirical_fpr, marker='.', linestyle='-', label="Generic BF", color="r")
        ax.legend()
        ax.title.set_text("%s, tau: %d" % (test_name, group1_label))
        ax.set_ylabel("Empirical FPR")
        ax.set_xscale("log")
        ax.set_xlabel("Table Size (bytes)")
        plt.savefig(f"laptop_benchmarks/{test_name}/tau-{group1_label}-log.png")
        plt.clf()
        plt.close()
