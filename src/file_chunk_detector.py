import re
import fasttext
import tldextract


class FileChunkDetector:

    def __init__(self):
        self.model = None
        self.accepted_chars = 'qwertyuioplkjhgfdsazxcvbnm-0123456789'
        self.white_words = self.get_white_words("../input/white_words")
        self.tld_extractor = tldextract.TLDExtract()

    def get_white_words(self, file_path):
        white_words = {line.strip() for line in open(file_path).readlines()}
        white_words_for_text = set()
        for w in white_words:
            if len(w) > 4:
                white_words_for_text.add(w)

        return white_words

    def has_whitelist(self, domain_name):
        result = False
        substrings = {domain_name[i: j] for i in range(len(domain_name)) for j in range(i + 1, len(domain_name) + 1)}

        if len(self.white_words & substrings) > 0: result = True

        return result

    def ip_contain(self, text):
        result = False

        if "-" in text or "." in text:
            dmn_name = text.replace("ip", "")
            dmn_name = dmn_name.replace("ptr", "")
            dmn_name = dmn_name.replace("crawl", "")
            dmn_name = dmn_name.replace("nat", "")
            #spt = dmn_name.split("-")
            spt = re.split(r'-|\.', dmn_name)
            if "" in spt:
                spt.remove("")

            digit_count = 0
            nt_digit = 0
            for s in spt:
                if s.isdigit():
                    digit_count += 1
                else:
                    nt_digit += 1

            if digit_count == 4 or (digit_count > 2 and nt_digit == 1):
                result = True

        return result

    def get_model_result(self, query):

        if not self.model:
            self.model = fasttext.load_model("../input/dga_quantized_model.bin")

        result = False  # False means negative, True means positive

        if self.ip_contain(query):
            result = False

        elif len(query) < 6:
            result = False

        else:
            try:
                query_idna = query.encode("utf-8").decode("idna")
            except:
                query_idna = query

            if self.has_whitelist(query_idna):
                result = False

            else:

                pred = self.model.predict(" ".join(list(query)))
                category = pred[0][0].replace("__label__", "")

                score = pred[1][0] if category == "dga" else 1 - pred[1][0]
                result = True if score > 0.5 else False

        return result

    def get_features_for_file_chunk_detector(self, fqdns):
        features = {"positive": 0, "negative": 0, "positive_ratio": 0.0}

        file_chunk_queries = []
        for fqdn in fqdns:
            subdomain = self.tld_extractor(fqdn).subdomain
            clean_subdomain = subdomain.replace("tunnel", "")
            clean_subdomain = clean_subdomain.replace("char", "")
            clean_subdomain = clean_subdomain.replace("slash", "")
            clean_subdomain = clean_subdomain.replace("slashchar", "")
            file_chunk_queries.append(clean_subdomain)

        for file_chunk in file_chunk_queries:
            res_file_chunk_model = self.get_model_result(file_chunk)

            if res_file_chunk_model:
                features["positive"] += 1
            else:
                features["negative"] += 1

        features["positive_ratio"] = features["positive_ratio"] / len(fqdns) if len(fqdns) != 0 else 0.0

        return features


def main():
    file_chunk_detector = FileChunkDetector()
    print(file_chunk_detector.get_model_result("1.1.1.1"))


if __name__ == "__main__":
    main()
