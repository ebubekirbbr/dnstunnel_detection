import argparse
import subprocess
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk, BulkIndexError
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

class PreprocessClass:

    def __init__(self):
        self.index_name = "tunnel_features_paper"

    def file2es(self, file_path, es_host):
        es = Elasticsearch(
            [es_host],  # Replace with your server address
        )

        mapping = {
            "mappings": {
                "dynamic": "strict",
                "_routing": {
                             "required": True
                },
              "properties": {
                "company_uid(masked)": {
                  "type": "keyword"
                },
                "dataset_version": {
                  "type": "keyword"
                },
                "dns_type": {
                  "type": "keyword"
                },
                "domain": {
                  "type": "keyword"
                },
                "fqdn": {
                  "type": "keyword"
                },
                "response": {
                  "type": "keyword"
                },
                "status": {
                  "type": "keyword"
                },
                  "is_tunnel_data": {
                      "type": "bool"
                  },
                "time": {
                  "type": "date"
                },
                "tld": {
                  "type": "keyword"
                }
              }
            }
                }

        if not es.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} does not exist. Creating it...")
            es.indices.create(index=self.index_name, body=mapping)
            print(f"Index {self.index_name} created with mapping.")
        else:
            print(f"Index {self.index_name} already exists.")

        # Settings
        batch_size = 100000  # Number of documents to fetch per batch or insert per batch

        # Function to import CSV data into Elasticsearch
        print(f"Reading CSV file {file_path}...")
        df = pd.read_csv(file_path)
        df = df.replace({np.nan: None})
        pbar = tqdm(ascii=True, desc="file2es", total=(len(df)//batch_size)+1)
        actions = []

        for i, record in df.iterrows():
            action = {
                "_index": self.index_name,
                "_source": {
                    "company_uid(masked)": record["company_uid(masked)"],
                    "fqdn": record["fqdn"],
                    "domain": record["domain"],
                    "tld": record["tld"],
                    "dns_type": record["dns_type"],
                    "time": record["time"],
                    "status": record["status"],

                }
            }
            actions.append(action)

            # Send in batches
            if len(actions) >= batch_size:
                try:
                    bulk(es, actions)
                except BulkIndexError as e:
                    print("Some documents failed to index.")
                    for error in e.errors:
                        print(error)

                pbar.update(1)
                actions = []

        # Insert any remaining records
        if actions:
            try:
                bulk(es, actions)
            except BulkIndexError as e:
                print("Some documents failed to index.")
                for error in e.errors:
                    print(error)

            print(f"Inserted final {len(actions)} records.")

        print("Import completed.")

    def es2file(self, file_path, es_host):
        es = Elasticsearch(
            [es_host],  # Replace with your server address
        )

        header_order = [line.strip() for line in open("../input/features.txt", "r").readlines()]
        # Fetch documents
        print("Fetching documents...")
        batch_size = 10000
        # Use scan to efficiently retrieve all documents

        bquery = {"query": {"bool": {"must": []}},
                  "_source": {"excludes": ["domain", "interval", "gte", "company", "version"]
        }}
        #bquery = {"query": {"bool": {"must": []}}}

        results = scan(
            es,
            index=self.index_name,
            query=bquery,
            scroll='2m',
            size=batch_size,  # Number of documents to fetch per batch

        )

        # Track batch and document counters
        batch_counter = 0
        doc_counter = 0
        first_batch = True
        pbar = tqdm(ascii=True, desc="es2file")
        # Open CSV file once
        with open(file_path, mode='w', encoding='utf-8-sig', newline='') as f:

            batch_docs = []

            for res in results:
                if doc_counter % batch_size == 0 and doc_counter != 0:
                    batch_counter += 1
                    df_batch = pd.DataFrame(batch_docs)

                    if first_batch:
                        df_batch.to_csv(f, index=False, columns=header_order)
                        first_batch = False
                    else:
                        df_batch.to_csv(f, index=False, header=False, columns=header_order)

                    pbar.update(1)
                    batch_docs = []

                source = res['_source']

                batch_docs.append(source)
                doc_counter += 1

            # Write any remaining documents
            if batch_docs:
                df_batch = pd.DataFrame(batch_docs)
                if first_batch:
                    df_batch.to_csv(f, index=False, columns=header_order)
                else:
                    df_batch.to_csv(f, index=False, header=False, columns=header_order)
                print(f"Final batch written to CSV.")

        print(f"Completed. {file_path} has been created.")

    def prep_dataset_for_train(self, file_path):
        df = pd.read_csv(file_path)

        # Stratified Split: label dağılımını koruyarak ayır
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,  # %20 test, %80 train
            stratify=df['label'],  # label dağılımını koru
            random_state=42  # tekrar üretilebilirlik için
        )

        # Sonuçları kaydet
        train_df.to_csv("../dataset/train.csv", index=False)
        test_df.to_csv("../dataset/test.csv", index=False)

        print("saved.")

    def prep_dataset_for_train_gen(self, path):

        """subprocess.call(f"head -1 {path}/dataset.csv > {path}/header.csv", shell=True)

        subprocess.call(f"cat {path}/dataset.csv | grep safe > {path}/safe.csv", shell=True)
        subprocess.call(f"cat {path}/dataset.csv | grep dnstunnel > {path}/dnstunnel.csv", shell=True)

        subprocess.call(f"shuf {path}/dnstunnel.csv -o {path}/dnstunnel_shuffled.csv", shell=True)

        subprocess.call(f"shuf {path}/safe.csv -o {path}/safe_shuffled.csv", shell=True)
        subprocess.call(f"shuf {path}/dnstunnel.csv -o {path}/dnstunnel_shuffled.csv", shell=True)
        #line_count_safe = int(subprocess.check_output(f"wc -l {path}/safe_shuffled.csv").decode("utf-8").encode("utf-8"))
        #line_count_dnstunnel = int(subprocess.check_output(f"wc -l {path}/dnstunnel_shuffled.csv").decode("utf-8").encode("utf-8"))"""
        line_count_safe = 37447179
        line_count_dnstunnel = 31135

        # %80 satır sayısını hesapla
        satir_sayisi_80_safe = line_count_safe * 80 // 100
        satir_sayisi_80_dnstunnel = line_count_dnstunnel * 80 // 100

        subprocess.call(f"head -n {satir_sayisi_80_safe} {path}/safe_shuffled.csv > {path}/train_tmp.csv", shell=True)
        subprocess.call(f"head -n {satir_sayisi_80_dnstunnel} {path}/dnstunnel_shuffled.csv >> {path}/train_tmp.csv", shell=True)

        subprocess.call(f"tail -n +{satir_sayisi_80_safe+1} {path}/safe_shuffled.csv > {path}/test_tmp.csv", shell=True)
        subprocess.call(f"tail -n +{satir_sayisi_80_dnstunnel+1} {path}/dnstunnel_shuffled.csv >> {path}/test_tmp.csv", shell=True)

        subprocess.call(f"shuf {path}/train_tmp.csv -o {path}/train_tmp_shuffled.csv", shell=True)
        subprocess.call(f"shuf {path}/test_tmp.csv -o {path}/test_tmp_shuffled.csv", shell=True)

        subprocess.call(f"cp {path}/header.csv {path}/train.csv", shell=True)
        subprocess.call(f"cat {path}/train_tmp_shuffled.csv >> {path}/train.csv", shell=True)

        subprocess.call(f"cp {path}/header.csv {path}/test.csv", shell=True)
        subprocess.call(f"cat {path}/test_tmp_shuffled.csv >> {path}/test.csv", shell=True)

        subprocess.call(f"rm -f train_tmp.csv test_tmp.csv train_tmp_shuffled.csv test_tmp_shuffled.csv", shell=True)










def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function", required=False, help='function name', type=str, default="main")
    parser.add_argument("--file_path", required=False, help='file name', type=str, default="../dataset/dataset.csv")
    parser.add_argument("--es_host", required=False, help='es_host', type=str, default="http://localhost:9200")
    parser.add_argument("--es_index", required=False, help='es index name', type=str, default="tunnel_dataset")
    args = parser.parse_args()

    return args


def main():
    preprocessor = PreprocessClass()
    args = argument_parsing()
    if args.function == "file2es":
        preprocessor.file2es(args.file_path, args.es_host)
    elif args.function == "es2file":
        preprocessor.es2file(args.file_path, args.es_host)
    elif args.function == "prep_dataset":
        preprocessor.prep_dataset_for_train_gen(args.file_path)
    else:
        print("enter a function")


if __name__ == "__main__":
    main()
