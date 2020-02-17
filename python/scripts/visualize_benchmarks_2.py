import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3 as sql
import pandas as pd


fixture_data = pd.read_csv("laptop_urls_normalized.csv", index_col=False)

# fixture_data["fpr"] = fixture_data.inv_max_fpr.astype(float)**-1
# columns = ["function", "projected_eles", "inv_max_fpr", "actual_fpr", "num_hashes", "table_size", "num_gen_eles", "len_gen_eles", "iterations", "real_time", "cpu_time"]
columns = ["function", "actual_fpr", "num_hashes", "table_size", "num_gen_eles", "len_gen_eles", "iterations", "real_time", "cpu_time"]
fixture_data = fixture_data[columns]
# print(fixture_data)


string_query = fixture_data[(fixture_data.function.str.contains("TestBloomFilterStringQuery"))]
# string_insertion = fixture_data[(fixture_data.function.str.contains("TestBloomFilterStringInsertion"))]
# int_insertion = fixture_data[(fixture_data.function.str.contains("TestBloomFilterIntInsertion"))]
int_query = fixture_data[(fixture_data.function.str.contains("TestBloomFilterIntQuery"))]

queries = [string_query, int_query]
querie_name = ["string_query", "int_query"]
sort_by = ["projected_eles", "fpr", "pos_eles", "ele_length"]
sort_by = ["projected_eles", "fpr", "pos_eles", "ele_length"]
time_aggregate = ["real_time", "cpu_time"]
colors = ["b", "c", "g", "y", "r", "m", "k"]
unique_lengths = fixture_data.len_gen_eles.unique()
unique_lengths.sort()
# print(unique_lengths)
colormap = dict(zip(unique_lengths, colors[: len(unique_lengths)]))

for ind, query in enumerate(queries): 
    for group1_label, group1 in query.groupby("table_size"):
        g = group1.sort_values("actual_fpr")
        # if(len(g.actual_fpr.unique()) == 1):
        #     continue
        for agg in time_aggregate:
            g["time_aggregate"] = g[agg] / g["num_gen_eles"]
            fig, ax = plt.subplots()
            for group2_label, group2 in g.groupby("len_gen_eles"):
                    ax.plot(group2.actual_fpr, group2.time_aggregate, marker='o', linestyle='', label=group2_label)
            ax.legend()
            ax.title.set_text("q: %s, tablesize: %d" % (querie_name[ind], group1_label))
            ax.set_ylabel("%s/item" % agg)
            plt.savefig("laptop_benchmarks/urls/%s/%s-q-%s-tsize-%d.png" % (querie_name[ind], agg, querie_name[ind], group1_label))
            plt.clf()
            plt.close()

        # break
    for group1_label, group1 in query.groupby("num_hashes"):
        g = group1.sort_values("actual_fpr")
        # if(len(g.actual_fpr.unique()) == 1):
        #     continue
        for agg in time_aggregate:
            g["time_aggregate"] = g[agg] / g["num_gen_eles"]
            fig, ax = plt.subplots()
            for group2_label, group2 in g.groupby("len_gen_eles"):
                    ax.plot(group2.actual_fpr, group2.time_aggregate, marker='o', linestyle='', label=group2_label)
            ax.legend()
            ax.title.set_text("q: %s, numhashes: %d" % (querie_name[ind], group1_label))
            ax.set_ylabel("%s/item" % agg)
            plt.savefig("laptop_benchmarks/urls/%s/%s-q-%s-numHashes-%d.png" % (querie_name[ind], agg, querie_name[ind], group1_label))
            plt.clf()
            plt.close()

        # break
    for group1_label, group1 in query.groupby(["num_hashes", "table_size"]):
        g = group1.sort_values("actual_fpr")
        # if(len(g.actual_fpr.unique()) == 1):
        #     continue
        for agg in time_aggregate:
            g["time_aggregate"] = g[agg] / g["num_gen_eles"]
            fig, ax = plt.subplots()
            for group2_label, group2 in g.groupby("len_gen_eles"):
                    ax.plot(group2.actual_fpr, group2.time_aggregate, marker='o', linestyle='', label=group2_label)
            ax.legend()
            ax.title.set_text("q: %s, n: %d t: %d" % (querie_name[ind], *group1_label))
            ax.set_ylabel("%s/item" % agg)
            plt.savefig("laptop_benchmarks/urls/%s/%s-q-%s-nh-%d-size-%d-nt.png" % (querie_name[ind], agg, querie_name[ind], *group1_label))
            plt.clf()
            plt.close()

        # break
            # # plt.show()
