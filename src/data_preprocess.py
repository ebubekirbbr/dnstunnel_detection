import argparse
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

    def prep_dataset_for_train_gen(self, file_path):
        # Sadece label sütununu oku
        labels = pd.read_csv(file_path, usecols=['label'])

        # Stratified split index'leri al
        train_idx, test_idx = train_test_split(
            labels.index,
            test_size=0.2,
            stratify=labels['label'],
            random_state=42
        )

        train_idx = set(train_idx)
        test_idx = set(test_idx)

        reader = pd.read_csv(file_path, chunksize=100_000)

        train_out = open("../dataset/train.csv", "w")
        test_out = open("../dataset/test.csv", "w")

        first_chunk = True
        total_rows_read = 0

        for chunk in tqdm(reader):
            chunk_len = len(chunk)
            chunk_idx = range(total_rows_read, total_rows_read + chunk_len)

            # hangi satırlar test/train
            chunk['row_id'] = list(chunk_idx)

            train_chunk = chunk[chunk['row_id'].isin(train_idx)]
            test_chunk = chunk[chunk['row_id'].isin(test_idx)]

            # İlk yazımda header yaz, sonra ekle
            train_chunk.drop(columns=["row_id"]).to_csv(train_out, mode='a', index=False, header=first_chunk)
            test_chunk.drop(columns=["row_id"]).to_csv(test_out, mode='a', index=False, header=first_chunk)

            first_chunk = False
            total_rows_read += chunk_len

        train_out.close()
        test_out.close()
        print("saved.")



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
