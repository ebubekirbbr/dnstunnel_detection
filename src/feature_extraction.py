import argparse
import json
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from multiprocessing import Process
from file_chunk_detector import FileChunkDetector
from elasticsearch.helpers import BulkIndexError


class FeatureExtractor:

    def __init__(self, es_host):
        self.index_name = "tunnel_dataset"
        self.index_features = "tunnel_features_paper"
        self.es_host = es_host

    def get_domain_features(self, es, interval, company):
        domains_info = []

        body = {
            "query": {
                "bool": {
                    "must": [{"range": {"time": {"gte": interval["gte"],
                                                 "lt": interval["lt"]}}},
                             {"term": {"company_uid(masked)": company}},
                             ],
                    "must_not": []
                }
            },
            "aggregations": {
                "domains": {
                        "terms": {"field": "domain", "size": 10000},
                        "aggs": {
                            "subd": {"cardinality": {"field": "fqdn"}},
                            "unique_responses": {"cardinality": {"field": "responses"}},
                            "status": {"terms": {"field": "status", "size": 2}},
                            "unique_dns_types": {"cardinality": {"field": "dns_type"}},
                            "dns_types": {"terms": {"field": "dns_type", "size": 20}},
                            "fqdns": {"terms": {"field": "fqdn", "size": 1000000}}
                        }
                    }
            },
            "size": 0
        }

        res = es.search(index=self.index_name, query=body["query"], routing=company, size=0, aggregations=body["aggregations"])

        for domain_bucket in res["aggregations"]["domains"]["buckets"]:

            dns_types = {"A": 0, "AAAA": 0, "TXT": 0, "SOA": 0, "NS": 0, "MX": 0, "DS": 0, "CNAME": 0, "HTTPS": 0, "OTHER": 0}
            status = {"success": 0, "fail": 0}
            fqdns = []

            for f_bucket in domain_bucket["fqdns"]["buckets"]:
                fqdns.append(f_bucket["key"])

            for s_bucket in domain_bucket["status"]["buckets"]:
                if s_bucket["key"] == "success":
                    status["success"] = s_bucket["doc_count"]
                elif s_bucket["key"] == "fail":
                    status["fail"] = s_bucket["doc_count"]

            for dt_buckets in domain_bucket["dns_types"]["buckets"]:
                if dt_buckets["key"] in ["A", "AAAA", "TXT", "SOA", "NS", "MX", "DS", "CNAME", "HTTPS"]:
                    dns_types[dt_buckets["key"]] = dt_buckets["doc_count"]
                else:
                    dns_types["OTHER"] += dt_buckets["doc_count"]

            domains_info.append({"domain": domain_bucket["key"],
                                 "fqdn_count": domain_bucket["subd"]["value"],
                                 "unique_response": domain_bucket["unique_responses"]["value"],
                                 "hits": domain_bucket["doc_count"],
                                 "unique_dns_types": domain_bucket["unique_dns_types"]["value"],
                                 "dns_types": dns_types,
                                 "status": status,
                                 "fqdns": fqdns,
                                 "company": company,
                                 "domain_data_portion": round(domain_bucket["doc_count"] / res["hits"]["total"]["value"], 6)
                                 }
                                )

        if len(domains_info) > 0:
            print("{} - domain count: {}  new iteration has started. gte: {}, lt: {}".format(company, len(domains_info), interval["gte"], interval["lt"]))
        return domains_info

    def get_date_intervals_every(self, start_date, end_date, window, interval):
        intervals = []

        # Başlangıç ve bitiş tarihlerini datetime formatına çevir
        start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
        end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")

        # Tarih listesini oluştur
        date_list = [start_date]

        # interval'e göre tarihleri listele
        while date_list[-1] + timedelta(minutes=interval) < end_date:
            date_list.append(date_list[-1] + timedelta(minutes=interval))

        # Aralıkları oluştur
        for x in date_list:
            intervals.append({
                "lt": x.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gte": (x - timedelta(minutes=int(window))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "window": window,
                "period": interval
            })

        return intervals

    def get_feature_for_other_companies(self, es, interval, company, domain):
        features = {
            "fqdn_count": 0,
            "fqdn_hit_ratio": 0,
            "hits": 0,
            "unique_company": 0,
            "fail_ratio": 0,
        }

        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"domain": domain}},
                        {"range": {"time": {"gte": interval["gte"], "lt": interval["lt"]}}}
                    ],
                    "must_not": [
                        {"term": {"company_uid(masked)": company}}
                    ]
                }
            },
            "aggregations": {
                "fqdn_count": {"cardinality": {"field": "fqdn"}},
                "unique_company": {"cardinality": {"field": "company_uid(masked)"}},
                "status": {"terms": {"field": "status"}}
            }
        }
        res_window = es.search(index=self.index_name, query=body["query"], routing=company, size=0, aggregations=body["aggregations"])

        features["fqdn_count"] = res_window["aggregations"]["fqdn_count"]["value"]
        features["unique_company"] = res_window["aggregations"]["unique_company"]["value"]
        features["hits"] = res_window["hits"]["total"]["value"]
        features["fqdn_hit_ratio"] = round(features["hits"] / features["fqdn_count"], 3) if features["fqdn_count"] > 0 else 0.0

        for bucket in res_window["aggregations"]["status"]["buckets"]:
            if bucket["key"] == "fail":
                features["fail_ratio"] = bucket["doc_count"] / features["hits"]

        return features

    def parallize_operation(self, fnc, pr_count, data):

        part = len(data) // pr_count
        procs = []

        for p in range(0, pr_count):
            start = p * part
            end = (p + 1) * part
            proc = Process(target=fnc, args=(p, pr_count, data[start:end]))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

    def job_main(self, p, pr, intervals):
        """
        :param p: process number
        :param pr: total process count
        :param intervals: time windows that will calculate features {'gte': '2025-02-27T00:00:00Z', 'lt': '2025-02-27T00:00:00Z'}
        :return: None
        """

        es = Elasticsearch([self.es_host])
        companies = [line.strip() for line in open("../input/companies.txt", "r").readlines()]
        tunnel_domains = json.loads(open("../input/tunnel_domains.json", "r").read())
        file_chunk_detector = FileChunkDetector()

        for interval in tqdm(intervals, ascii=True, desc=f"intervals - p:{p}", position=p):

            for company in companies:
                domains_info = self.get_domain_features(es, interval, company)
                es_data = []

                for d_info in domains_info:

                    _id = f"{interval['gte']}_{company}_{d_info['domain']}"
                    response = es.exists(index=self.index_features, id=_id, routing=company)

                    if response:  # pass if features calculated before
                        continue

                    last_24 = (datetime.strptime(interval["gte"], '%Y-%m-%dT%H:%M:%SZ') - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')
                    interval_last24 = {"gte": last_24, "lt": interval["gte"]}  # select last 24 before interval gte

                    features_for_other_comp_window = self.get_feature_for_other_companies(es, interval, company, d_info["domain"])
                    features_for_other_comp_last24 = self.get_feature_for_other_companies(es, interval_last24, company, d_info["domain"])

                    file_chunk_features = file_chunk_detector.get_features_for_file_chunk_detector(d_info["fqdns"])

                    sorted_items = sorted(d_info["dns_types"].items(), key=lambda x: x[1], reverse=True)
                    top_one = sorted_items[0]

                    feature = {
                        "domain": d_info["domain"],
                        "label": "dnstunnel" if d_info["domain"] in tunnel_domains else "safe",
                        "company": d_info["company"],
                        "hits": d_info["hits"],
                        "domain_data_portion": d_info["domain_data_portion"],
                        "fqdn_count": d_info["fqdn_count"],
                        "fqdn_hit_ratio": round(d_info["hits"] / d_info["fqdn_count"] , 3),
                        "dns_request_fail_count": d_info["status"]["fail"],
                        "dns_request_fail_ratio": round(d_info["status"]["fail"] / d_info["hits"], 3),
                        "dns_request_success_count": d_info["status"]["success"],
                        "dns_request_success_ratio": round(d_info["status"]["success"] / d_info["hits"], 3),
                        "unique_dns_types": d_info["unique_dns_types"],
                        "dns_type_A_count": d_info["dns_types"]["A"],
                        "dns_type_A_ratio": round(d_info["dns_types"]["A"] / d_info["hits"], 3),
                        "dns_type_AAAA_count": d_info["dns_types"]["AAAA"],
                        "dns_type_AAAA_ratio": round(d_info["dns_types"]["AAAA"] / d_info["hits"], 3),
                        "dns_type_TXT_count": d_info["dns_types"]["TXT"],
                        "dns_type_TXT_ratio": round(d_info["dns_types"]["TXT"] / d_info["hits"], 3),
                        "dns_type_SOA_count": d_info["dns_types"]["SOA"],
                        "dns_type_SOA_ratio": round(d_info["dns_types"]["SOA"] / d_info["hits"], 3),
                        "dns_type_NS_count": d_info["dns_types"]["NS"],
                        "dns_type_NS_ratio": round(d_info["dns_types"]["NS"] / d_info["hits"], 3),
                        "dns_type_MX_count": d_info["dns_types"]["MX"],
                        "dns_type_MX_ratio": round(d_info["dns_types"]["MX"] / d_info["hits"], 3),
                        "dns_type_DS_count": d_info["dns_types"]["DS"],
                        "dns_type_DS_ratio": round(d_info["dns_types"]["DS"] / d_info["hits"], 3),
                        "dns_type_CNAME_count": d_info["dns_types"]["CNAME"],
                        "dns_type_CNAME_ratio": round(d_info["dns_types"]["CNAME"] / d_info["hits"], 3),
                        "dns_type_HTTPS_count": d_info["dns_types"]["HTTPS"],
                        "dns_type_HTTPS_ratio": round(d_info["dns_types"]["HTTPS"] / d_info["hits"], 3),
                        "dns_type_OTHER_count": d_info["dns_types"]["OTHER"],
                        "dns_type_OTHER_ratio": round(d_info["dns_types"]["OTHER"] / d_info["hits"], 3),
                        "top1_records_count": d_info["dns_types"][top_one[0]],
                        "top1_records_ratio": round(d_info["dns_types"][top_one[0]] / d_info["hits"], 3),

                        "other_comp_last_24_fqdn_count": features_for_other_comp_last24["fqdn_count"],
                        "other_comp_last_24_fqdn_hit_ratio": float(features_for_other_comp_last24["fqdn_hit_ratio"]),
                        "other_comp_last_24_hits": features_for_other_comp_last24["hits"],
                        "other_comp_last_24_unique_company": features_for_other_comp_last24["unique_company"],
                        "other_comp_last_24_fail_ratio": float(features_for_other_comp_last24["fail_ratio"]),

                        "other_comp_window_fqdn_count": features_for_other_comp_window["fqdn_count"],
                        "other_comp_window_fqdn_hit_ratio": float(features_for_other_comp_window["fqdn_hit_ratio"]),
                        "other_comp_window_hits": features_for_other_comp_window["hits"],
                        "other_comp_window_unique_company": features_for_other_comp_window["unique_company"],
                        "other_comp_window_fail_ratio": float(features_for_other_comp_window["fail_ratio"]),

                        "file_chunk_detection_positive": file_chunk_features["positive"],
                        "file_chunk_detection_negative": file_chunk_features["negative"],
                        "file_chunk_detection_positive_ratio": float(file_chunk_features["positive_ratio"]),
                        "interval": interval

                    }

                    _id = f"{interval['gte']}_{company}_{d_info['domain']}"
                    es_data.append({"_index": self.index_features, "_source": feature, "_id": _id})

                try:
                    helpers.bulk(es, es_data)
                except BulkIndexError as e:
                    print(f"Bulk index failed. Errors: {e.errors}")

    def main(self, pr):

        intervals = self.get_date_intervals_every(
            start_date="2025-02-27T00:00:00Z",
            end_date="2025-02-27T23:59:59Z",
            window=5,
            interval=1
        )

        if pr == 1:
            self.job_main(0, 1, intervals)
        else:
            self.parallize_operation(self.job_main, pr, intervals)


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function", required=False, help='function name', type=str, default="main")
    parser.add_argument("--file_path", required=False, help='file name', type=str, default="../dataset/dataset.csv")
    parser.add_argument("--es_host", required=False, help='es_host', type=str, default="http://localhost:9200")
    parser.add_argument("--es_index", required=False, help='es index name', type=str, default="tunnel_dataset")
    parser.add_argument("--process", required=False, help='process count', type=int, default=1)
    args = parser.parse_args()

    return args


def main():
    args = argument_parsing()
    preprocessor = FeatureExtractor(args.es_host)
    preprocessor.main(args.process)


if __name__ == "__main__":
    main()