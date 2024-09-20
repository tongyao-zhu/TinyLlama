import json
import random
import fasttext
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import pdb


# Load Qwen tokenizer
def load_qwen_tokenizer(model_name='Qwen/Qwen2-0.5B'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


# Sample positive and negative samples
def sample_data_with_labels(positive_jsonl_file, negative_jsonl_file, sample_num, pool_size, max_length=512):
    positive_seed_set = []
    negative_pool_set = []

    # Load positive samples
    with open(positive_jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            text = data['text'].replace('\n', ' ')
            words = text.split()
            truncated_text = ' '.join(words[:max_length])
            positive_seed_set.append(truncated_text)

    # Load a large pool of negative samples
    with open(negative_jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            text = data['text'].replace('\n', ' ')
            words = text.split()
            truncated_text = ' '.join(words[:max_length])
            negative_pool_set.append(truncated_text)

    # Ensure the pool size doesn't exceed the available data
    pool_size = min(len(negative_pool_set), pool_size)
    full_negative_pool = random.sample(negative_pool_set, pool_size)
    holdout_set = full_negative_pool[:1000]
    full_negative_pool = full_negative_pool[1000:]
    # Sample a smaller subset of negatives for training
    sample_num = min(len(positive_seed_set), sample_num)
    sample_positive = random.sample(positive_seed_set, sample_num)
    sample_negative = random.sample(full_negative_pool, sample_num)

    # Add labels
    labeled_positive = [("__label__positive", text) for text in sample_positive]
    labeled_negative = [("__label__negative", text) for text in sample_negative]
    holdout_set = [("__label__negative", text) for text in holdout_set]

    return labeled_positive, labeled_negative, holdout_set, full_negative_pool


# SAVE train file
def save_data_to_file(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write(line + '\n')


# use Qwen tokenizer to split tokens
def preprocess_sampled_data(labeled_data, tokenizer, max_length=512):
    formatted_data = []
    for label, text in tqdm(labeled_data, desc=f'Processing data'):
        inputs = tokenizer(text, truncation=True, padding=True, max_length=max_length)
        token_ids = inputs['input_ids']
        token_ids_str = " ".join(f"w{str(token_id)}" for token_id in token_ids)
        formatted_data.append(f"{label} " + token_ids_str)
    return formatted_data


# Train fasttext
def train_fasttext_with_pretrained_vectors(train_data_file, vec_file, model_save_path):
    print('Training FastText model with pre-trained vectors...')
    model = fasttext.train_supervised(
        input=train_data_file,
        epoch=3,
        lr=0.1,
        dim=896,
        wordNgrams=3,
        minCount=3,
        pretrainedVectors=vec_file
    )
    # Save model
    model.save_model(model_save_path)
    print(f'Model saved to {model_save_path}')
    return model


# Collect possible positive samples
def get_positive_example(negative_data, labels, scores, max_length=512, threshold_score=0.5, top_n=100):
    relabel_positive_set = []
    example_ids = []
    for i, text in tqdm(enumerate(negative_data), desc='Get positive example...'):
        if i < len(labels) and i < len(scores):  # Ensure index is within bounds
            if labels[i] != '__label__negative' and scores[i] > threshold_score:
                relabel_positive_set.append(text)
                example_ids.append(i)
    # pdb.set_trace()
    # 对 __label__positive 的数据排序，获取分数最高的 Top N
    positive_indices = [i for i in range(len(labels)) if labels[i] == '__label__positive']
    sorted_positive_indices = sorted(positive_indices, key=lambda x: scores[x], reverse=True)
    top_goods = [negative_data[i] for i in sorted_positive_indices[:top_n]]  # Top N good examples (highest scores)

    # 对 __label__negative 的数据排序，获取分数最低的 Top N
    negative_indices = [i for i in range(len(labels)) if labels[i] == '__label__negative']
    sorted_negative_indices = sorted(negative_indices, key=lambda x: scores[x])
    top_bads = [negative_data[i] for i in sorted_negative_indices[:top_n]]  # Top N bad examples (lowest scores)

    return example_ids, relabel_positive_set, top_goods, top_bads
    # return example_ids, relabel_positive_set,top_goods,top_bads


# 保存状态
def save_state(positive_seed_set, negative_seed_set, relabel_positive_set, iteration):
    state = {
        'positive_seed_set': positive_seed_set,
        'negative_seed_set': negative_seed_set,
        'iteration': iteration
    }
    # Define the directory and file path
    dir_path = 'ckpt'
    file_path = os.path.join(dir_path, 'split_into_5_model_qwen_id_state.json')

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Write to the file
    with open(file_path, 'w') as file:
        json.dump(state, file, ensure_ascii=False)

    file_path = os.path.join(dir_path, 'relable_file_test.json')
    # Write to the file
    relabeled_data = {
        'relabel_data': relabel_positive_set,
    }
    with open(file_path, 'a+') as file:
        json.dump(relabeled_data, file, ensure_ascii=False)


# Check
def predict_and_check_correctness(model, tokenizer, labeled_data, max_length=512):
    correct_predictions = 0
    total_samples = len(labeled_data)

    for label, text in tqdm(labeled_data, desc="Checking correctness"):
        inputs = tokenizer(text, truncation=True, padding=True, max_length=max_length)
        token_ids = inputs['input_ids']
        token_ids_str = " ".join(f"w{str(token_id)}" for token_id in token_ids)

        prediction = model.predict(token_ids_str)
        predicted_label = prediction[0][0]

        if predicted_label == label:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples * 100
    print(f'Correct predictions: {correct_predictions}/{total_samples}')
    print(f'Accuracy: {accuracy:.2f}%')


def main(positive_jsonl_file, negative_jsonl_file, sample_num, threshold, pool_size, vec_file='qwen_embeddings_ids.vec',
         model_name='Qwen/Qwen2-0.5B', max_length=512):
    print('Loading Qwen tokenizer...')
    tokenizer = load_qwen_tokenizer(model_name)

    print('Sampling positive and negative data...')
    labeled_positive_data, labeled_negative_data, holdout_negative_pool, full_negative_pool = sample_data_with_labels(
        positive_jsonl_file, negative_jsonl_file, sample_num, pool_size, max_length)
    # First three of them are like '__LABEL__ corpus', last is corpus
    print('Processing sampled data...')
    positive_data = preprocess_sampled_data(labeled_positive_data, tokenizer, max_length)
    negative_data = preprocess_sampled_data(labeled_negative_data, tokenizer, max_length)

    epoch = 0
    while epoch < 6:

        train_set = positive_data + negative_data
        random.shuffle(train_set)

        # Ensure the directory exists
        model_dir = 'split_into_5_model_qwen'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        print(f'Epoch: {epoch}')

        print('Training FastText model...')
        model_save_path = os.path.join(model_dir, f'iterative_model_{epoch}_epoch.bin')

        train_data_file = 'train_data.txt'
        save_data_to_file(train_set, train_data_file)
        model = train_fasttext_with_pretrained_vectors(train_data_file, vec_file, model_save_path)

        scores = []
        labels = []
        for line in tqdm(full_negative_pool, desc='Predicting labels...'):  # Use the full relabel pool
            if len(line.split(' ', 1)) >= 2:  # Ensure there are enough tokens
                label, score = model.predict(line.split(' ', 1)[1])
                scores.append(score[0])
                labels.append(label[0])

        print(f"Length of full_negative_pool: {len(full_negative_pool)}")
        print(f"Length of labels: {len(labels)}")
        print(f"Length of scores: {len(scores)}")

        positive_example_ids, relabel_positive_set, top_goods, top_bads = get_positive_example(full_negative_pool,
                                                                                               labels, scores,
                                                                                               max_length)

        labeled_positive_data += [("__label__positive", text) for text in relabel_positive_set]
        labeled_negative_data = [("__label__negative", text) for i, text in enumerate(full_negative_pool) if
                                 i not in positive_example_ids]

        print(f'Number of relabel examples: {len(relabel_positive_set)}')

        print('Saving state...')
        save_state(labeled_positive_data, labeled_negative_data, relabel_positive_set, epoch)

        positive_data += top_goods
        negative_data += top_bads

        # Check
        original_train_data = labeled_positive_data[:50] + labeled_negative_data[:50]

        print('Verifying model accuracy on original training set (50+50 samples)...')
        predict_and_check_correctness(model, tokenizer, original_train_data, max_length)
        print('Verifying model accuracy on original holdout...')
        predict_and_check_correctness(model, tokenizer, holdout_negative_pool, max_length)
        # Update the full_negative_pool by removing the relabeled positive examples
        full_negative_pool = [text for i, text in enumerate(full_negative_pool) if i not in positive_example_ids]
        epoch += 1
        if len(relabel_positive_set) < threshold:
            break


if __name__ == "__main__":
    sample_num = 500
    threshold = 10
    pool_size = 50000  # Large pool size for relabeling
    vec_file = 'qwen_embeddings_ids.vec'
    # vec_file='/nfs-share/multilingual_synthetic_dataset/cc.th.300.vec'
    main('/nfs-share/multilingual_synthetic_dataset/merge_cos_thai.jsonl', './merge_raw_thai.jsonl', sample_num,
         threshold, pool_size, vec_file)
