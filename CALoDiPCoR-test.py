import numpy as np
from Client import Client
from Server import Server


def load_dataset(path, max_clients=None, max_reports=None):
    data = np.loadtxt(path, delimiter=',', dtype=float)
    data = data.astype(int)
    if max_clients is not None and max_reports is not None:
        total = max_clients * max_reports
        data = data[:total]
        data = data.reshape(max_clients, max_reports)
    return data


def main():

    m = 200
    n = 100        
    h = 50        
    thresh = 5
    coef = 10
    epsilon_1 = 1
    epsilon_inf = 2
    dataset_path = './datasets/NJ2021.csv' 

    values_matrix = load_dataset(dataset_path, max_clients=n, max_reports=h)

    reports = []
    consumed_budget = []

    for i in range(n):
        budget = coef * epsilon_inf
        client = Client(epsilon_inf, epsilon_1, m, thresh, budget,
                        fixthreshold=0.8, corr=True)

        values = values_matrix[i, :]
        client_report = client.report(values)
        reports.append(client_report)
        consumed_budget.append(budget - client.budget)

    server = Server(epsilon_inf, epsilon_1, m)
    mu_hat = server.estimate_mean(reports)

    all_values = values_matrix.flatten()
    unique_vals, counts = np.unique(all_values, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    mu_real = float(np.sum(unique_vals * probs))

    estimation_error = abs(mu_real - mu_hat)
    avg_consumed_budget = float(np.mean(consumed_budget))

    print("Real Mean: ", mu_real)
    print("Estimated Mean (LoDiPCoR-CAL): ", mu_hat)
    print("Estimation Error: ", estimation_error)
    print("Average Consumed Budget: ", avg_consumed_budget)

if __name__ == "__main__":
    main()

