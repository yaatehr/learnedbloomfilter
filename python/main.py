import os
import argparse
import pickle
import string
from data_intake import data_loader, processing
import multiprocessing
from datetime import datetime
num_processes = multiprocessing.cpu_count()
from classifier import train, export_model

def query_google_sb(args, use_checkpoint=True):
    extractor = processing.WebCrawlExtractor(args)
    url_classifier = processing.UrlClassifier(args)
    # unclassified_url_path = os.path.join(args.root, 'input/clean_dedup_urls.txt')
    time_now = datetime.now().strftime("%m-%d-%H:%M:%S")
    unclassified_url_path = os.path.join(
        args.root, "input/webcrawl_unique_len_12_prefixes.txt"
    )
    classified_url_path = os.path.join(args.root, "input/classified_crawl_%s.txt" % time_now)
    cached_iteration = 0
    metadata_root = os.path.join(args.root, "python/url_gen")

    if not os.path.exists(metadata_root):
        os.makedirs(metadata_root)

    checkpoint_path = os.path.join(metadata_root, "url_class_%s.pkl" % (time_now))
    metadata_path = os.path.join(metadata_root, "url_class_%s.txt" % (time_now))
    print(os.path.basename(metadata_path))
    date_of_run = datetime.strptime(os.path.basename(metadata_path)[10:].split(".")[0], "%m-%d-%H:%M:%S")
    most_recent_run_path = None

    if use_checkpoint and os.listdir(metadata_root):
        paths = os.walk(metadata_root)
        dates = []
        for root, dirs, files in paths:
            for i, path in enumerate(files):
                date_of_run = datetime.strptime(path[10:].split(".")[0], "%m-%d-%H:%M:%S")
                dates.append((date_of_run, i))
            dates = sorted(dates, key=lambda x: x[0], reverse=True)
            most_recent_run_path = os.path.join(metadata_root, files[dates[0][1]][:-3] + "txt")
            break
        print("most recent path found: ", most_recent_run_path)

    if most_recent_run_path and os.path.exists(most_recent_run_path):
        with open(most_recent_run_path, 'rb') as fp: #update the url paths to append
            cached_iteration, unclassified_url_path, classified_url_path = pickle.load(fp)
            print("Successfully loaded run metadata from %s -\n iter: %d\n unclass_path: %s\n class_path: %s" % (most_recent_run_path, cached_iteration, unclassified_url_path, classified_url_path))

    with open(unclassified_url_path, "r+") as ip:
        batch = []
        fast_forwarding = most_recent_run_path != None
        for i, line in enumerate(ip):

            if fast_forwarding:
                if i < cached_iteration:
                    continue
                else:
                    fast_forwarding = False

            if not fast_forwarding and i != 0 and i % 100000 == 0:
                with open(checkpoint_path, "wb") as fp:
                    pickle.dump(url_classifier, fp)
                    print("url classifier checkpoint at %s" % checkpoint_path)
                with open(metadata_path, "wb") as fp:
                    pickle.dump((i, unclassified_url_path, classified_url_path), fp)
                    print("url classifier metadata at %s" % metadata_path)

            if not fast_forwarding and i != 0 and i % 500 == 0:
                malicious, benign, stopflag = url_classifier.query(batch)
                if stopflag:
                    with open(metadata_path, "wb") as fp:
                        i-= 500*args.query_timeout
                        pickle.dump((i, unclassified_url_path, classified_url_path), fp)
                        print("STOPPING RUN DUE TO ERROR TIMEOUT:\n\t url classifier metadata at %s" % metadata_path)
                        return

                if not os.path.exists(classified_url_path):
                    fp = open(classified_url_path, "w+")
                    fp.write("isMalicious url\n")
                    fp.close()

                with open(classified_url_path, "a") as fp:
                    for url in malicious:
                        fp.write("1 " + url + "\n")
                    for url in benign:
                        fp.write("0 " + url + "\n")
                batch = []
            batch.append(line.strip())

    with open(checkpoint_path, "wb") as fp:
        pickle.dump(url_classifier, fp)
        print("url classifier run COMPLETED\n checkpoint at %s" % checkpoint_path)

def train_model(args):
    texts, labels, tokens, number_of_classes, sample_weights = data_loader.load_data(
        args
    )
    dataset = data_loader.EncodedDataset(texts, labels, args, tokens=tokens)
    dataset.__getitem__(1)
    train.run(args)

def load_dataset_from_shallalist(args):
    urls_by_category_path = os.path.join(args.root, "python/scripts/url_load_backup.pkl")
    with open(urls_by_category_path, 'rb') as fp:
        urls_by_category = pickle.load(fp)
    dataset = data_loader.EncodedStringLabelDataset(args, urls_by_category)
    return dataset
    # print(dataset.__getitem__(0))

def main_loop(args):
    # query_google_sb(args)
    # train_model(args)
    # dataset = load_dataset_from_shallalist(args)
    train.run(args)
    # export_model.export_blank_model(args)
    # extractor.process_crawl()
    if args.debug:
        print("DEBUG: end main_loop")


# NOTE CITATION infrastructure for model training from https://github.com/ahmedbesbes/character-based-cnn
if __name__ == "__main__":
    URL_ALPHABET = string.ascii_letters + string.digits + "_.~-" + ":/?#[]@!$&'()*+,;="
    URL_DELIM = ":/?#.&"
    DEFAULT_ALPHABET = ( #TODO fix to add spaces to en embedding (just have vals of 0?)
        "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{}"
    )
    DIRECTORY_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    print("running main loop from \n  DIRECTORY ROOT: ", DIRECTORY_ROOT)
    parser = argparse.ArgumentParser("Classifier trainer")
    parser.add_argument("--root", type=str, default=DIRECTORY_ROOT)
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(DIRECTORY_ROOT, "input/randlabelurls.txt"),
        # default=os.path.join(DIRECTORY_ROOT, "input/classified_web_crawl_urls.txt"),
    )  # TODO (finetuning - change this to point to the true training data)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--label_column", type=str, default="Label")
    parser.add_argument("--text_column", type=str, default="Text")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument(
        "--sep", type=str, default="\s+"
    )  # TODO update if we need to seperate by commas
    parser.add_argument("--num_text_processing_threads", type=int, default=2)
    parser.add_argument("--use_char_encoding", type=int, default=0)

    # parser.add_argument('--steps', nargs='+', default=['lower'])
    # parser.add_argument('--group_labels', type=int, default=1, choices=[0, 1])
    # parser.add_argument('--ignore_center', type=int, default=1, choices=[0, 1])
    # parser.add_argument('--label_ignored', type=int, default=None)
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--balance", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_sampler", type=int, default=0, choices=[0, 1])

    parser.add_argument("--alphabet", type=str, default=DEFAULT_ALPHABET)
    parser.add_argument("--number_of_characters", type=int, default=69)
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=124)
    parser.add_argument("--url_delimeters", type=str, default=URL_DELIM)

    parser.add_argument("--min_word_count", type=int, default=1)
    parser.add_argument("--embedding_window", type=int, default=3)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--embedding_path", type=str, default=os.path.join(DIRECTORY_ROOT, 'input/glove.6B.50d-char.txt'))

    # training params
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dropout_input", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--class_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--focal_loss", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--scheduler", type=str, default="step", choices=["clr", "step"]
    )
    parser.add_argument("--min_lr", type=float, default=1.7e-3)
    parser.add_argument("--max_lr", type=float, default=1e-2)
    parser.add_argument("--stepsize", type=float, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early_stopping", type=int, default=0, choices=[0, 1])
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--checkpoint", type=int, choices=[0, 1], default=1)

    # Rnn params
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--num_hidden_layers", type=int, default=1)  # TODO deprecate?
    parser.add_argument("--bidirectional", type=int, default=0, choices=[0, 1])

    # logging params
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--log_f1", type=int, default=1, choices=[0, 1])
    parser.add_argument("--flush_history", type=int, default=0, choices=[0, 1])
    parser.add_argument("--output", type=str, default="./modelsaves/")
    parser.add_argument("--model_name", type=str, default="")

    parser.add_argument("--dry_run", type=int, default=0, choices=[0,1])

    # parser.add_argument("--in_set_labels", type=list, default=["sports","travel","humor","martialarts","wellness","restaurants"])
    parser.add_argument("--in_set_labels", type=list, default=['porn', 'models', 'education', 'lingerie'])
    parser.add_argument("--drop_out_set", type=int, default=1, choices=[0,1])             #TODO Deprecate this hack?
    parser.add_argument("--use_string_labels", type=int, default=1, choices=[0,1])
    parser.add_argument("--use_word2vec_encoding", type=int, default=1, choices=[0,1])
    # parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--query_timeout", type=int, default=3)

    parser.add_argument("--debug", type=bool, default=True)

    args = parser.parse_args()
    args.embedding_size_bytes = (len(args.alphabet) + 1) * args.embedding_size * 32 # bytes per float in numpy float32

    main_loop(args)



