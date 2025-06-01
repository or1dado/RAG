from pyserini.index import IndexReader
import numpy as np
import math


class QppPreRetrival:
    def __init__(self, index_path):
        self.index_reader = IndexReader(index_path)
        self.ignore_terms = ["on", "an", 'it', 'at', 'be', 'or', 'will', 'and', 'in', 'no', 'a.', 'is']
        self.num_docs = self.index_reader.stats()['documents']

    def text2tokens(self, qtext):
        return self.index_reader.analyze(qtext)

    def _IDF(self, term):
        df, cf = self.index_reader.get_term_counts(term, analyzer=None)

        if df == 0:
            return 0.0
        else:
            return math.log2(self.num_docs / df)

    def _SCQ(self, term):
        df, cf = self.index_reader.get_term_counts(term, analyzer=None)

        if cf == 0 or term in self.ignore_terms:
            return 0.0
        else:
            part_A = 1 + math.log2(cf)
            part_B = self._IDF(term)

        return part_A * part_B

    def avg_max_sum_SCQ(self, qtokens):
        scq = []
        for t in qtokens:
            scq.append(self._SCQ(t))
        return {"mean": np.mean(scq), "max": max(scq), "sum": sum(scq)}

    def avg_max_sum_std_IDF(self, qtokens):
        v = []
        for t in qtokens:
            v.append(self._IDF(t))
        return {"mean": np.mean(v), "max": max(v), "sum": sum(v), "std": np.std(v)}

    def _VAR(self, t):
        # one query token, multiple docs containing it
        postings_list = self.index_reader.get_postings_list(t, analyzer=None)

        if postings_list == None:
            return 0.0, 0.0
        else:
            tf_array = np.array([posting.tf for posting in postings_list])
            tf_idf_array = np.log2(1 + tf_array) * self._IDF(t)

            return {"var": np.var(tf_idf_array), "std": np.std(tf_idf_array)}

    def avg_max_sum_VAR(self, qtokens):
        v = []
        for t in qtokens:
            v.append(self._VAR(t)["var"])

        return {"mean": np.mean(v), "max": max(v), "sum": sum(v)}


if __name__ == "__main__":
    qpp_pre_retrival = QppPreRetrival(r'/lv_local/home/or.dado/PycharmProjects/RAG/create_index/downloads/data/wikipedia_split/w100_luence_index')

    qtext = "Where in England was Dame Judi Dench born?"
    qtokens = qpp_pre_retrival.text2tokens(qtext)

    # SCQ
    scq_score = qpp_pre_retrival.avg_max_sum_SCQ(qtokens)

    # IDF
    idf_score = qpp_pre_retrival.avg_max_sum_std_IDF(qtokens)

    # VAR
    var_score = qpp_pre_retrival.avg_max_sum_VAR(qtokens)

    1==1
