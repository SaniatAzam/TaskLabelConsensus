import pandas as pd
import os
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 1000)
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# By Sa Adat Azam Saniat - 260827887.
# COMP 596 Final Project


is_book_df = pd.read_csv("binary-classification/Crowdsourced Amazon Sentiment/transformed_dataset_is_book.csv")
is_negative_df = pd.read_csv("binary-classification/Crowdsourced Amazon Sentiment/transformed_dataset_is_negative.csv")
sentiment_popularity_df = pd.read_csv("binary-classification/Sentiment popularity - AMT/transformed_dataset.csv")
weather_sentiment_df = pd.read_csv("multi-class-classification/Weather Sentiment - AMT/transformed_dataset.csv")

# Additional
blue_bird_df = pd.read_csv("binary-classification/Blue Birds/transformed_dataset.csv")
recognizing_textual_entailment = pd.read_csv("binary-classification/Recognizing Textual Entailment/transformed_dataset.csv")



weather_sentiment_df = weather_sentiment_df[['workerID', 'taskID', 'response', 'timeSpent']]
sentiment_popularity_df = sentiment_popularity_df[['workerID', 'taskID', 'response', 'timeSpent']]
negative_reviews_df = is_negative_df[['workerID', 'taskID', 'response']]
book_reviews_df = is_book_df[['workerID', 'taskID', 'response']]

# Additional
blue_bird_df = blue_bird_df[['workerID', 'taskID', 'response']]
rte_df = recognizing_textual_entailment[['workerID', 'taskID', 'response']]



def simple_consensus_method(df, name):
    checked_taskIDs = []

    for index, row in df.iterrows():
        if row['taskID'] in checked_taskIDs:
            continue
        checked_taskIDs.append(row['taskID'])
        task_df = df.loc[df['taskID'] == row['taskID']]
        consensus_response = task_df['response'].value_counts().idxmax()
        df.loc[df['taskID'] == row['taskID'], 'consensus'] = consensus_response
        consensus_count = task_df.loc[task_df['response'] == consensus_response].shape[0]
        task_size = task_df.shape[0]
        consensus_confidence = consensus_count / task_size
        df.loc[df['taskID'] == row['taskID'], 'consensusCount'] = consensus_count
        df.loc[df['taskID'] == row['taskID'], 'totalWorkers'] = task_df.shape[0]
        df.loc[df['taskID'] == row['taskID'], 'consensusConfidence'] = consensus_confidence

    filename = f'test_data/{name}_consensus.csv'
    if os.path.isfile(filename):
        df.to_csv(filename, mode='w', index=False)
    else:
        df.to_csv(filename, mode='a', index=False)
    return df



def worker_experience(df, name):
    checked_workerID = []
    w_df = pd.DataFrame(columns=['workerID', 'totalTasks'])

    for index, row in df.iterrows():
        if row['workerID'] in checked_workerID:
            continue
        checked_workerID.append(row['workerID'])
        worker_tasks = df.loc[df['workerID'] == row['workerID'], 'taskID'].nunique()
        w_df.loc[len(w_df)] = [row['workerID'], worker_tasks]
    w_df.to_csv(f'test_data/{name}_worker_experience.csv', index=False)
    return w_df




weather_consensus = simple_consensus_method(weather_sentiment_df, "weather")
weather_worker_exp = worker_experience(weather_consensus, "weather")

sentiment_popularity_consensus = simple_consensus_method(sentiment_popularity_df, "sentiment_popularity")
sentiment_popularity_worker_exp = worker_experience(sentiment_popularity_consensus, "sentiment_popularity")

negative_reviews_consensus = simple_consensus_method(negative_reviews_df, "negative_reviews")
negative_reviews_worker_exp = worker_experience(negative_reviews_consensus, "negative_reviews")

book_reviews_consensus = simple_consensus_method(book_reviews_df, "book_reviews")
book_reviews_worker_exp = worker_experience(book_reviews_consensus, "book_reviews")

# Additional
blue_bird_consensus = simple_consensus_method(blue_bird_df, "blue_bird")
blue_bird_worker_exp = worker_experience(blue_bird_consensus, "blue_bird")

rte_consensus = simple_consensus_method(rte_df, "rte")
rte_worker_exp = worker_experience(rte_consensus, "rte")




def accuracy_vs_workerExp(worker_exp, consensus, name, m_one=False, conversion=False):

    if not m_one:
        worker_data = worker_exp.groupby('workerID')['totalTasks'].max().reset_index()

        consensus_confidence = consensus.groupby('workerID')['consensusConfidence'].mean().reset_index()

        if conversion:
            worker_data['workerID'] = worker_data['workerID'].astype('str')
            consensus_confidence = consensus_confidence['workerID'].astype('str')

        data = worker_data.merge(consensus_confidence, on='workerID', how='left')

        data.columns = ['workerID', 'totalTasks', 'consensusConfidenceMean']

        lowess = sm.nonparametric.lowess(data['consensusConfidenceMean'], data['totalTasks'])
        plt.scatter(data['totalTasks'], data['consensusConfidenceMean'])
        plt.plot(lowess[:, 0], lowess[:, 1], 'g-', linewidth=2, label="Lowess")

        fit = np.polyfit(data['totalTasks'], data['consensusConfidenceMean'], 2)

        x = np.linspace(data['totalTasks'].min(), data['totalTasks'].max(), 100)
        y = np.polyval(fit, x)
        plt.plot(x, y, 'r-', linewidth=2, label="Polynomial Fit")

        plt.xlabel('Worker Experience (Total Tasks)')
        plt.ylabel("Mean Consensus Confidence")
        plt.title(f"Work Experience (Total Tasks) - Mean Accuracy of {name}")
        plt.legend()
        plt.savefig(f"taskplots/{name}_task_m2")
        plt.show()

    else:
        task_workers = consensus.groupby('taskID')['workerID'].unique().reset_index()

        task_data = task_workers.apply(lambda row: pd.Series({
            'taskID': row['taskID'],
            'totalTasks': worker_exp.loc[worker_exp['workerID'].isin(row['workerID'])]['totalTasks'].sum()
        }), axis=1)


        task_data = task_data.merge(consensus[['taskID', 'consensusConfidence']], on='taskID', how='left')


        lowess = sm.nonparametric.lowess(task_data['consensusConfidence'], task_data['totalTasks'])
        plt.scatter(task_data['totalTasks'], task_data['consensusConfidence'])
        plt.plot(lowess[:, 0], lowess[:, 1], 'g-', linewidth=2, label="Lowess")


        fit = np.polyfit(task_data['totalTasks'], task_data['consensusConfidence'], 2)

        x = np.linspace(task_data['totalTasks'].min(), task_data['totalTasks'].max(), 100)
        y = np.polyval(fit, x)
        plt.plot(x, y, 'r-', linewidth=2, label="Polynomial Fit")

        plt.xlabel('Total Tasks of Workers on Task')
        plt.ylabel('Task Consensus Confidence')
        plt.title(f"Accuracy of predicted labels w.r.t. the experience of the workers of {name}")
        plt.legend()
        plt.savefig(f"taskplots/{name}_task_m1")
        plt.show()


accuracy_vs_workerExp(weather_worker_exp, weather_consensus, "Weather Sentiment", m_one=True)
accuracy_vs_workerExp(weather_worker_exp, weather_consensus, "Weather Sentiment", m_one=False)

accuracy_vs_workerExp(sentiment_popularity_worker_exp, sentiment_popularity_consensus, "Sentiment Popularity", m_one=True)
accuracy_vs_workerExp(sentiment_popularity_worker_exp, sentiment_popularity_consensus, "Sentiment Popularity", m_one=False)


accuracy_vs_workerExp(negative_reviews_worker_exp, negative_reviews_consensus, "isNegative", m_one=True)
accuracy_vs_workerExp(negative_reviews_worker_exp, negative_reviews_consensus, "isNegative", m_one=False)

accuracy_vs_workerExp(book_reviews_worker_exp, book_reviews_consensus, "isBook", m_one=True)
accuracy_vs_workerExp(book_reviews_worker_exp, book_reviews_consensus, "isBook", m_one=False)

# Additional
accuracy_vs_workerExp(blue_bird_worker_exp, blue_bird_consensus, "Blue Bird", m_one=True, conversion=False)
accuracy_vs_workerExp(blue_bird_worker_exp, blue_bird_consensus, "Blue Bird", m_one=False, conversion=False)

accuracy_vs_workerExp(rte_worker_exp, rte_consensus,  "RTE", m_one=True, conversion=False)
accuracy_vs_workerExp(rte_worker_exp ,rte_consensus, "RTE", m_one=False, conversion=False)

def confidence_pies(df, threshold=0.5, name=None):
    above_threshold = df[df['consensusConfidence'] >= threshold]

    above_count = len(above_threshold['taskID'].unique())

    above_percentage = above_count / len(df['taskID'].unique()) * 100


    below_percentage = 100 - above_percentage


    labels = ['Above Threshold', 'Below Threshold']
    sizes = [above_percentage, below_percentage]
    colors = ['#ADD8E6', '#FFA500']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f'{name} Task Consensus Confidence. Threshold = {threshold}')
    plt.show()


def confidence_percentages(df, threshold=0.5, name=None):
    above_threshold = df[df['consensusConfidence'] >= threshold]
    above_count = len(above_threshold['taskID'].unique())
    above_percentage = above_count / len(df['taskID'].unique()) * 100
    below_percentage = 100 - above_percentage

    return above_percentage, below_percentage

def create_thrshold_graphs(df, name):
    above_t = []
    below_t = []

    for t in np.arange(0.0, 1.01, 0.01):
        above, below = confidence_percentages(df, threshold=t, name=name)
        above_t.append(above)
        below_t.append(below)

    plt.plot(np.arange(0.0, 1.01, 0.01), above_t, label='Above threshold')
    plt.plot(np.arange(0.0, 1.01, 0.01), below_t, label='Below or equal threshold')

    plt.axvline(x=0.5, linestyle='--', color='gray', label='x=0.5')

    y_above = above_t[50]
    y_below = below_t[50]

    plt.text(0.5, y_above, f"{y_above:.2f}", ha='center', va='bottom', color='black')
    plt.text(0.5, y_below, f"{y_below:.2f}", ha='center', va='top', color='black')
    plt.plot([0.5], [above_t[50]], marker='o', markersize=5, color="blue")
    plt.plot([0.5], [below_t[50]], marker='o', markersize=5, color="orange")

    plt.xlabel('Threshold')
    plt.ylabel('Percentage')
    plt.title(f'Percentage above/below threshold for {name}')
    plt.legend()
    plt.savefig(f'plots/{name}_percentages')

    plt.show()



create_thrshold_graphs(weather_consensus, "Weather Sentiment")
create_thrshold_graphs(sentiment_popularity_consensus, "Sentiment Popularity")
create_thrshold_graphs(book_reviews_consensus, "isBook - CAS")
create_thrshold_graphs(negative_reviews_consensus, "isNegative - CAS")

# Additional
create_thrshold_graphs(blue_bird_consensus, "Blue Bird")
create_thrshold_graphs(rte_consensus, "RTE")

def time_graphs(df, name):
    task_stats = df.groupby('taskID').agg({'timeSpent': 'mean', 'consensusConfidence': 'mean'}).reset_index()

    lowess = sm.nonparametric.lowess(task_stats['consensusConfidence'], task_stats['timeSpent'])
    plt.scatter(task_stats['timeSpent'], task_stats['consensusConfidence'])
    plt.plot(lowess[:, 0], lowess[:, 1], 'g-', linewidth=2, label='Lowess')

    fit = np.polyfit(task_stats['timeSpent'], task_stats['consensusConfidence'], 2)

    plt.scatter(task_stats['timeSpent'], task_stats['consensusConfidence'])

    x = np.linspace(task_stats['timeSpent'].min(), task_stats['timeSpent'].max(), 100)
    y = np.polyval(fit, x)
    plt.plot(x, y, 'r-', linewidth=2, label='Polynomial Fit')

    plt.xlabel('Average Time Spent (seconds)')
    plt.ylabel('Task Consensus Confidence')
    plt.title(f'Average Time Spent VS Consesus Confidence for {name}')
    plt.legend()
    plt.savefig(f'timeplots/{name}_tp')
    plt.show()


time_graphs(weather_consensus, "Weather Sentiment")
time_graphs(sentiment_popularity_consensus, "Sentiment Popularity")
