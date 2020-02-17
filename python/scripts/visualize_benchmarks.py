import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3 as sql
import pandas as pd


fixture_data = pd.read_csv("fixture1_normalized.csv", index_col=False)

fixture_data["fpr"] = fixture_data.inv_max_fpr.astype(float)**-1
columns = ["function", "projected_eles", "fpr", "pos_eles", "ele_length", "iterations", "real_time", "cpu_time"]
fixture_data = fixture_data[columns]


string_query = fixture_data[(fixture_data.function.str.contains("TestBloomFilterStringQuery"))]
string_insertion = fixture_data[(fixture_data.function.str.contains("TestBloomFilterStringInsertion"))]
int_insertion = fixture_data[(fixture_data.function.str.contains("TestBloomFilterIntInsertion"))]
int_query = fixture_data[(fixture_data.function.str.contains("TestBloomFilterIntQuery"))]

queries = [string_query, string_insertion, int_insertion, int_query]
querie_name = ["string_query", "string_insertion", "int_insertion", "int_query"]
sort_by = ["projected_eles", "fpr", "pos_eles", "ele_length"]
sort_by = ["projected_eles", "fpr", "pos_eles", "ele_length"]
time_aggregate = ["iterations", "real_time", "cpu_time"]

for ind, query in enumerate(queries): 
    for ele_length, group in query.groupby("ele_length"):
        g = group.sort_values("fpr")
        for agg in time_aggregate:
            g["time_aggregate"] = g[agg] / g["pos_eles"]
            g.plot(x="fpr", y=agg, kind="scatter", title="q: %s, maxlen: %d" % (querie_name[ind], ele_length)).set_ylabel("%s/item" % agg)
            plt.savefig("%s-q-%s-mlen-%d-2.png" % (agg, querie_name[ind], ele_length))
            # # plt.show()
