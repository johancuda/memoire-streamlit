import pandas as pd
import statistics

def topicsToCsv(topic_model, run_i):
    # Get general topic info
    topic_info = topic_model.get_topic_info()

    # Expand each topic with keywords
    topic_details = []
    for topic_id in topic_info["Topic"]:
        if topic_id != -1:  # Exclude outliers
            keywords = topic_model.get_topic(topic_id)
            topic_details.append({
                "Topic ID": topic_id,
                "Topic Name": topic_info.loc[topic_info["Topic"] == topic_id, "Name"].values[0],
                "Keywords": ", ".join([word[0] for word in keywords])
            })

    # Convert to DataFrame
    df_topics = pd.DataFrame(topic_details)

    df_topics.to_csv(f"topics_0{run_i}.csv", index=False, sep=";")


def topic_diversity(topic_model, topk=10):
    all_topics = topic_model.get_topics()
    topic_words = [words[:topk] for _, words in [
        (topic_id, [word for word, _ in topic_words])
        for topic_id, topic_words in all_topics.items()
        if topic_id != -1
    ]]
    
    unique_words = set([word for topic in topic_words for word in topic])
    total_words = len(topic_words) * topk
    
    return len(unique_words) / total_words


def corpus_mean_length(liste):

    len_mean = sum(map(len, liste)) / len(liste)
    print(f"Corpus length: {len(liste)}")

    text_list = set(liste)
    text_list = list(text_list)

    print(f"Corpus length without doubles: {len(text_list)}")

    return len_mean