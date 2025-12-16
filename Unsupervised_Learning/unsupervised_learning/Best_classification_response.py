# In other files
import API

# Call function to get the best cluster
result = API.get_best_cluster_from_csv('clusters/cluster_statistics.csv')

# Use the results
print(f"Best cluster file: {result['cluster_file']}")
print(f"Selection reason: {result['selection_reason']}")
print(f"SR probability: {result['prob_SR']}")
print(f"YM probability: {result['prob_YM']}")
print(f"SR≥9 sample count: {result['SR_ge_9']}")
print(f"YM suitable sample count: {result['YM_100_2000']}")
print(f"API status: {result['api_status']}")