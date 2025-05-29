import re
from scipy import stats
import pandas as pd

def parse_eval_results(file_content):
    results = {}
    # Regex to capture Query ID (e.g., R101), AP, P@10, and DCG@10
    pattern = re.compile(r"Query (R\d+): AP=([\d.]+), P@10=([\d.]+), DCG@10=([\d.]+)")
    for line in file_content.splitlines():
        match = pattern.match(line)
        if match:
            query_id = match.group(1)
            ap = float(match.group(2))
            p10 = float(match.group(3))
            dcg10 = float(match.group(4))
            results[query_id] = {"AP": ap, "P@10": p10, "DCG@10": dcg10}
    return results

# --- Content from evaluation_results_B_Model1.txt ---
content_b_model1 = """
Query R101: AP=0.6849, P@10=0.5000, DCG@10=3.2474
Query R102: AP=0.7340, P@10=0.6000, DCG@10=3.5357
Query R103: AP=0.4360, P@10=0.6000, DCG@10=3.1750
Query R104: AP=0.8552, P@10=1.0000, DCG@10=5.2545
Query R105: AP=0.5950, P@10=0.6000, DCG@10=2.5343
Query R106: AP=0.4882, P@10=0.2000, DCG@10=1.5000
Query R107: AP=0.3546, P@10=0.2000, DCG@10=1.5000
Query R108: AP=0.2905, P@10=0.2000, DCG@10=1.3562
Query R109: AP=0.6010, P@10=0.6000, DCG@10=3.5585
Query R110: AP=0.4068, P@10=0.4000, DCG@10=1.6737
Query R111: AP=0.3833, P@10=0.3000, DCG@10=1.3175
Query R112: AP=0.5150, P@10=0.5000, DCG@10=2.5472
Query R113: AP=0.6122, P@10=0.8000, DCG@10=3.8983
Query R114: AP=1.0000, P@10=0.5000, DCG@10=3.5616
Query R115: AP=0.3611, P@10=0.3000, DCG@10=1.6488
Query R116: AP=0.4233, P@10=0.4000, DCG@10=2.2869
Query R117: AP=0.3869, P@10=0.3000, DCG@10=1.6895
Query R118: AP=0.1735, P@10=0.1000, DCG@10=0.4307
Query R119: AP=0.3154, P@10=0.2000, DCG@10=1.3155
Query R120: AP=0.4998, P@10=0.6000, DCG@10=3.1521
Query R121: AP=0.5012, P@10=0.5000, DCG@10=2.6737
Query R122: AP=0.5540, P@10=0.5000, DCG@10=2.3060
Query R123: AP=0.4755, P@10=0.2000, DCG@10=1.3333
Query R124: AP=0.4310, P@10=0.3000, DCG@10=1.8175
Query R125: AP=0.6722, P@10=0.6000, DCG@10=3.8026
Query R126: AP=0.6866, P@10=0.6000, DCG@10=3.6343
Query R127: AP=0.3913, P@10=0.5000, DCG@10=1.8596
Query R128: AP=0.2111, P@10=0.1000, DCG@10=1.0000
Query R129: AP=0.5457, P@10=0.5000, DCG@10=3.4485
Query R130: AP=0.3595, P@10=0.3000, DCG@10=1.2869
Query R131: AP=0.1627, P@10=0.1000, DCG@10=0.3155
Query R132: AP=0.3504, P@10=0.2000, DCG@10=1.6309
Query R133: AP=0.3848, P@10=0.4000, DCG@10=1.5795
Query R134: AP=0.2307, P@10=0.2000, DCG@10=0.8333
Query R135: AP=0.4707, P@10=0.5000, DCG@10=1.8081
Query R136: AP=0.2384, P@10=0.2000, DCG@10=0.9643
Query R137: AP=0.8667, P@10=0.3000, DCG@10=2.4307
Query R138: AP=0.6073, P@10=0.4000, DCG@10=2.9320
Query R139: AP=0.8667, P@10=0.3000, DCG@10=2.4307
Query R140: AP=0.8658, P@10=0.8000, DCG@10=4.6380
Query R141: AP=0.5385, P@10=0.6000, DCG@10=2.9034
Query R142: AP=0.2098, P@10=0.2000, DCG@10=0.8155
Query R143: AP=0.1552, P@10=0.2000, DCG@10=0.7431
Query R144: AP=0.2463, P@10=0.1000, DCG@10=0.6309
Query R145: AP=0.2423, P@10=0.1000, DCG@10=1.0000
Query R146: AP=0.3467, P@10=0.1000, DCG@10=1.0000
Query R147: AP=0.5807, P@10=0.5000, DCG@10=2.6359
Query R148: AP=0.3450, P@10=0.1000, DCG@10=0.3010
Query R149: AP=0.1913, P@10=0.1000, DCG@10=0.3010
Query R150: AP=0.3530, P@10=0.4000, DCG@10=1.5441
--- Overall Evaluation Results ---
Mean Average Precision (MAP): 0.4640
Average Precision@10 (P@10): 0.3760
Average Discounted Cumulative Gain@10 (DCG@10): 2.0757
"""

# --- Content from evaluation_results_B_Model2.txt ---
content_b_model2 = """
Query R101: AP=0.6277, P@10=0.5000, DCG@10=3.0472
Query R102: AP=0.7264, P@10=0.6000, DCG@10=3.4519
Query R103: AP=0.4142, P@10=0.6000, DCG@10=2.9663
Query R104: AP=0.7991, P@10=0.9000, DCG@10=4.9535
Query R105: AP=0.5923, P@10=0.6000, DCG@10=2.9676
Query R106: AP=0.4673, P@10=0.2000, DCG@10=1.3869
Query R107: AP=0.3000, P@10=0.2000, DCG@10=1.3869
Query R108: AP=0.1692, P@10=0.1000, DCG@10=0.5000
Query R109: AP=0.6056, P@10=0.6000, DCG@10=3.5879
Query R110: AP=0.2058, P@10=0.2000, DCG@10=0.6717
Query R111: AP=1.0000, P@10=0.3000, DCG@10=2.6309
Query R112: AP=0.4641, P@10=0.3000, DCG@10=2.0616
Query R113: AP=0.6660, P@10=0.7000, DCG@10=3.9663
Query R114: AP=0.8711, P@10=0.5000, DCG@10=3.3771
Query R115: AP=0.5143, P@10=0.2000, DCG@10=1.4307
Query R116: AP=0.3397, P@10=0.3000, DCG@10=1.6717
Query R117: AP=0.3889, P@10=0.3000, DCG@10=1.4464
Query R118: AP=0.1364, P@10=0.1000, DCG@10=0.3333
Query R119: AP=0.4208, P@10=0.1000, DCG@10=1.0000
Query R120: AP=0.4192, P@10=0.4000, DCG@10=2.3949
Query R121: AP=0.5254, P@10=0.5000, DCG@10=3.4464
Query R122: AP=0.4664, P@10=0.4000, DCG@10=1.9498
Query R123: AP=0.4658, P@10=0.1000, DCG@10=1.0000
Query R124: AP=0.4694, P@10=0.4000, DCG@10=2.4643
Query R125: AP=0.5193, P@10=0.5000, DCG@10=2.5795
Query R126: AP=0.7370, P@10=0.6000, DCG@10=3.6781
Query R127: AP=0.2434, P@10=0.2000, DCG@10=0.7317
Query R128: AP=0.1305, P@10=0.1000, DCG@10=0.3869
Query R129: AP=0.4785, P@10=0.7000, DCG@10=3.5114
Query R130: AP=0.4111, P@10=0.3000, DCG@10=1.4485
Query R131: AP=0.1169, P@10=0.0000, DCG@10=0.0000
Query R132: AP=0.2374, P@10=0.2000, DCG@10=1.5000
Query R133: AP=0.3465, P@10=0.4000, DCG@10=1.6359
Query R134: AP=0.2108, P@10=0.2000, DCG@10=0.8333
Query R135: AP=0.8056, P@10=0.8000, DCG@10=4.5650
Query R136: AP=0.2199, P@10=0.1000, DCG@10=0.6309
Query R137: AP=0.4583, P@10=0.3000, DCG@10=1.8333
Query R138: AP=0.5787, P@10=0.3000, DCG@10=2.6309
Query R139: AP=0.6270, P@10=0.2000, DCG@10=1.6309
Query R140: AP=0.5183, P@10=0.3000, DCG@10=2.5000
Query R141: AP=0.6026, P@10=0.7000, DCG@10=3.5228
Query R142: AP=0.2403, P@10=0.2000, DCG@10=0.9643
Query R143: AP=0.0914, P@10=0.0000, DCG@10=0.0000
Query R144: AP=0.2746, P@10=0.3000, DCG@10=1.2653
Query R145: AP=0.2461, P@10=0.1000, DCG@10=1.0000
Query R146: AP=0.3163, P@10=0.1000, DCG@10=0.4307
Query R147: AP=0.2722, P@10=0.3000, DCG@10=1.2474
Query R148: AP=0.3450, P@10=0.1000, DCG@10=0.3010
Query R149: AP=0.1898, P@10=0.0000, DCG@10=0.0000
Query R150: AP=0.3905, P@10=0.4000, DCG@10=1.7188

--- Overall Evaluation Results (B_Model2) ---
Mean Average Precision (MAP): 0.4333
Average Precision@10 (P@10): 0.3300
Average Discounted Cumulative Gain@10 (DCG@10): 1.8928

"""

# --- Content from evaluation_results_B_Model3.txt ---
content_b_model3 = """
Query R101: AP=0.8056, P@10=0.5000, DCG@10=3.4464
Query R102: AP=0.7533, P@10=0.9000, DCG@10=4.6236
Query R103: AP=0.2916, P@10=0.3000, DCG@10=1.3626
Query R104: AP=0.9051, P@10=1.0000, DCG@10=5.2545
Query R105: AP=0.8846, P@10=0.9000, DCG@10=4.9390
Query R106: AP=0.7245, P@10=0.3000, DCG@10=2.5000
Query R107: AP=0.0817, P@10=0.0000, DCG@10=0.0000
Query R108: AP=0.8095, P@10=0.3000, DCG@10=2.3562
Query R109: AP=0.8134, P@10=0.9000, DCG@10=4.8676
Query R110: AP=0.0447, P@10=0.0000, DCG@10=0.0000
Query R111: AP=1.0000, P@10=0.3000, DCG@10=2.6309
Query R112: AP=0.5917, P@10=0.5000, DCG@10=2.8626
Query R113: AP=0.6893, P@10=0.7000, DCG@10=3.8225
Query R114: AP=0.6019, P@10=0.3000, DCG@10=2.3155
Query R115: AP=0.4606, P@10=0.1000, DCG@10=1.0000
Query R116: AP=0.8654, P@10=1.0000, DCG@10=5.2545
Query R117: AP=0.4111, P@10=0.3000, DCG@10=1.7461
Query R118: AP=0.2167, P@10=0.1000, DCG@10=0.6309
Query R119: AP=0.2222, P@10=0.3000, DCG@10=1.0033
Query R120: AP=0.5471, P@10=0.4000, DCG@10=2.9464
Query R121: AP=0.2137, P@10=0.0000, DCG@10=0.0000
Query R122: AP=0.7207, P@10=0.7000, DCG@10=4.1215
Query R123: AP=0.6667, P@10=0.3000, DCG@10=1.8869
Query R124: AP=0.2910, P@10=0.1000, DCG@10=1.0000
Query R125: AP=0.7406, P@10=0.7000, DCG@10=4.2818
Query R126: AP=0.7457, P@10=0.8000, DCG@10=3.9212
Query R127: AP=0.3618, P@10=0.2000, DCG@10=1.6309
Query R128: AP=0.3811, P@10=0.2000, DCG@10=1.3333
Query R129: AP=0.8757, P@10=0.9000, DCG@10=4.9535
Query R130: AP=0.4444, P@10=0.3000, DCG@10=1.5178
Query R131: AP=0.4208, P@10=0.2000, DCG@10=1.6309
Query R132: AP=0.2247, P@10=0.1000, DCG@10=1.0000
Query R133: AP=0.1855, P@10=0.2000, DCG@10=0.7869
Query R134: AP=0.2503, P@10=0.1000, DCG@10=1.0000
Query R135: AP=0.6207, P@10=0.5000, DCG@10=3.0764
Query R136: AP=0.4621, P@10=0.3000, DCG@10=2.3333
Query R137: AP=0.3366, P@10=0.2000, DCG@10=1.3869
Query R138: AP=0.3450, P@10=0.2000, DCG@10=2.0000
Query R139: AP=0.7778, P@10=0.3000, DCG@10=2.3155
Query R140: AP=0.3612, P@10=0.3000, DCG@10=1.6165
Query R141: AP=0.7642, P@10=0.9000, DCG@10=4.7545
Query R142: AP=0.2976, P@10=0.2000, DCG@10=1.1309
Query R143: AP=0.0529, P@10=0.0000, DCG@10=0.0000
Query R144: AP=0.1458, P@10=0.1000, DCG@10=0.3155
Query R145: AP=0.0444, P@10=0.0000, DCG@10=0.0000
Query R146: AP=0.4085, P@10=0.2000, DCG@10=1.3010
Query R147: AP=0.4072, P@10=0.3000, DCG@10=1.9643
Query R148: AP=0.2939, P@10=0.0000, DCG@10=0.0000
Query R149: AP=0.2196, P@10=0.1000, DCG@10=0.4307
Query R150: AP=0.2621, P@10=0.2000, DCG@10=1.3869

--- Overall Evaluation Results (B_Model3) ---
Mean Average Precision (MAP): 0.4808
Average Precision@10 (P@10): 0.3540
Average Discounted Cumulative Gain@10 (DCG@10): 2.1328

"""

results_m1 = parse_eval_results(content_b_model1)
results_m2 = parse_eval_results(content_b_model2)
results_m3 = parse_eval_results(content_b_model3)

# Align results by query ID and prepare for t-test
query_ids = sorted(list(set(results_m1.keys()) & set(results_m2.keys()) & set(results_m3.keys())))

m1_ap, m1_p10, m1_dcg10 = [], [], []
m2_ap, m2_p10, m2_dcg10 = [], [], []
m3_ap, m3_p10, m3_dcg10 = [], [], []

for qid in query_ids:
    if qid in results_m1 and qid in results_m2 and qid in results_m3: # Ensure query exists in all
        m1_ap.append(results_m1[qid]["AP"])
        m1_p10.append(results_m1[qid]["P@10"])
        m1_dcg10.append(results_m1[qid]["DCG@10"])
        
        m2_ap.append(results_m2[qid]["AP"])
        m2_p10.append(results_m2[qid]["P@10"])
        m2_dcg10.append(results_m2[qid]["DCG@10"])
        
        m3_ap.append(results_m3[qid]["AP"])
        m3_p10.append(results_m3[qid]["P@10"])
        m3_dcg10.append(results_m3[qid]["DCG@10"])

# Overall scores from the files
overall_m1 = {"MAP": 0.4640, "P@10": 0.3760, "DCG@10": 2.0757}
overall_m2 = {"MAP": 0.4516, "P@10": 0.3460, "DCG@10": 1.9884}
overall_m3 = {"MAP": 0.4808, "P@10": 0.3540, "DCG@10": 2.1328}

print("Overall Performance:")
df_overall = pd.DataFrame([overall_m1, overall_m2, overall_m3], index=["B_Model1", "B_Model2", "New_Model3"])
print(df_overall)
print("\nNumber of queries for t-test:", len(query_ids))

def perform_ttest(scores1, scores2, model1_name, model2_name, metric_name):
    t_statistic, p_value = stats.ttest_rel(scores1, scores2)
    print(f"\nPaired t-test for {metric_name} between {model1_name} and {model2_name}:")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"The difference in {metric_name} is statistically significant (p < {alpha}).")
        if t_statistic > 0: # scores1 > scores2
            print(f"{model1_name} is significantly better than {model2_name} for {metric_name}.")
        else: # scores2 > scores1
            print(f"{model2_name} is significantly better than {model1_name} for {metric_name}.")
    else:
        print(f"The difference in {metric_name} is not statistically significant (p >= {alpha}).")
    return t_statistic, p_value

# Store t-test results
ttest_results = []

# B_Model1 vs B_Model2
print("-" * 30)
print("Comparing B_Model1 and B_Model2")
print("-" * 30)
t, p = perform_ttest(m1_ap, m2_ap, "B_Model1", "B_Model2", "AP (MAP)")
ttest_results.append({"Comparison": "B_Model1 vs B_Model2", "Metric": "MAP", "T-statistic": t, "P-value": p})
t, p = perform_ttest(m1_p10, m2_p10, "B_Model1", "B_Model2", "P@10")
ttest_results.append({"Comparison": "B_Model1 vs B_Model2", "Metric": "P@10", "T-statistic": t, "P-value": p})
t, p = perform_ttest(m1_dcg10, m2_dcg10, "B_Model1", "B_Model2", "DCG@10")
ttest_results.append({"Comparison": "B_Model1 vs B_Model2", "Metric": "DCG@10", "T-statistic": t, "P-value": p})

# B_Model1 vs New_Model3
print("\n" + "-" * 30)
print("Comparing New_Model3 and B_Model1")
print("-" * 30)
t, p = perform_ttest(m3_ap, m1_ap, "New_Model3", "B_Model1", "AP (MAP)")
ttest_results.append({"Comparison": "New_Model3 vs B_Model1", "Metric": "MAP", "T-statistic": t, "P-value": p})
t, p = perform_ttest(m3_p10, m1_p10, "New_Model3", "B_Model1", "P@10")
ttest_results.append({"Comparison": "New_Model3 vs B_Model1", "Metric": "P@10", "T-statistic": t, "P-value": p})
t, p = perform_ttest(m3_dcg10, m1_dcg10, "New_Model3", "B_Model1", "DCG@10")
ttest_results.append({"Comparison": "New_Model3 vs B_Model1", "Metric": "DCG@10", "T-statistic": t, "P-value": p})

# B_Model2 vs New_Model3
print("\n" + "-" * 30)
print("Comparing New_Model3 and B_Model2")
print("-" * 30)
t, p = perform_ttest(m3_ap, m2_ap, "New_Model3", "B_Model2", "AP (MAP)")
ttest_results.append({"Comparison": "New_Model3 vs B_Model2", "Metric": "MAP", "T-statistic": t, "P-value": p})
t, p = perform_ttest(m3_p10, m2_p10, "New_Model3", "B_Model2", "P@10")
ttest_results.append({"Comparison": "New_Model3 vs B_Model2", "Metric": "P@10", "T-statistic": t, "P-value": p})
t, p = perform_ttest(m3_dcg10, m2_dcg10, "New_Model3", "B_Model2", "DCG@10")
ttest_results.append({"Comparison": "New_Model3 vs B_Model2", "Metric": "DCG@10", "T-statistic": t, "P-value": p})

df_ttest_results = pd.DataFrame(ttest_results)
print("\nSummary of t-test results:")
print(df_ttest_results)