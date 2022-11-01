import glob
import pandas
import json

def read_ukp_sent():
    data_path = "../../data/UKP_sentential_argument_mining/data"

    samples = pandas.concat([
        pandas.read_csv(file, delimiter="\t", quotechar="'")
        for file in glob.glob(data_path + "/*")
    ])
    samples = samples[['topic', 'sentence', 'annotation']]
    samples.rename(columns={"annotation":"label"}, inplace=True)
    return samples

def read_pstance():
    data_path = "../../data/PStance/"

    samples = pandas.concat([
        pandas.read_csv(data_path + "train.csv"),
        pandas.read_csv(data_path + "val.csv"),
        pandas.read_csv(data_path + "test.csv")
    ])

    samples.rename(columns={"Tweet": "sentence", "Stance":"label", "Target":"topic"}, inplace=True)
    return samples
    # samples["label"] = samples["label"].apply(lambda label: label_mapping[label])

def read_vast():
    # 0=con, 1=pro, 2=neutral
    data_path = "../../data/VAST/"
    samples = pandas.concat([
        pandas.read_csv(file)
        for file in glob.glob(data_path + "/*.txt")
    ])
    columns = ["text_s", "topic_str", "label"]
    samples = samples[columns]
    samples.rename(columns={"text_s": "sentence", "topic_str": "topic"}, inplace=True)
    label_map = {
        1:  'pro',
        0: 'con',
        2: 'neutral'
    }
    samples["label"] = samples["label"].apply(lambda label: label_map[label])
    return samples

def read_semevalt6():
    data_path = "../../data/SemEvalT6/"

    samples = pandas.concat([
        pandas.read_csv(data_path + "train.csv", sep=',', engine='python'),
        pandas.read_csv(data_path + "test.csv", sep=',', engine='python')
    ])
    columns = ["Tweet","Stance", "Target"]
    samples = samples[columns]
    samples.rename(columns={"Tweet": "sentence", "Stance":"label", "Target":"topic"}, inplace=True)
    return samples

def read_perspectrum():
    data_path = "../../data/perspectrum/"

    with open(data_path+"perspectrum_with_answers_v1.0.json","r") as claims_in, \
            open(data_path+"perspective_pool_v1.0.json", "r") as perspectives_in:

        # load files
        claims = json.load(claims_in)
        perspectives = json.load(perspectives_in)

        # lookup for perspective ids
        perspectives_dict = {}
        for p in perspectives:
            perspectives_dict[p['pId']] = p['text']

        # init
        X = []

        for claim in claims:
            for p_cluster in claim['perspectives']:
                cluster_label = p_cluster['stance_label_3']
                for pid in p_cluster['pids']:
                    X.append({'label': cluster_label, 'sentence': perspectives_dict[pid], 'topic': claim['text']})

    samples = pandas.DataFrame(X)
    return samples

def read_ibm_arg_q():
    # 1 (pro), -1 (con) and 0 (neutral)
    data_path = "../../data/IBM_Arg_Qual"

    samples = pandas.read_csv(data_path + "/arg_quality_rank_30k.csv")
    label_map = {
        1:  'pro',
        -1: 'con',
        0: 'neutral'
    }
    samples["sentence"] = samples["argument"]
    samples["label"] = samples["stance_WA"].apply(lambda label: label_map[label])

    samples = samples[["sentence", "topic", "label"]]
    return samples

def read_ibm_evi():
    #1 for evidence (either supporting or contesting the topic), 0 for non-evidence
    data_path = "../../data/IBMDebaterEvidenceSentences"

    samples = pandas.concat([
        pandas.read_csv(data_path + "/train.csv"),
        pandas.read_csv(data_path + "/test.csv")
    ])

    samples["sentence"] = samples["candidate"]

    del samples["candidate"]
    return samples

if __name__ == '__main__':
    datasets = {
        'IBM-Evidence': read_ibm_evi,
        'Perspectrum': read_perspectrum,
        'VAST': read_vast,
        'PStance': read_pstance,
        'IBM-Argument-Quality-Stance': read_ibm_arg_q,
        'SemEval 2016 Task 6': read_semevalt6,
        'UKP-Sentential Argument Mining': read_ukp_sent
    }
    for name, f in datasets.items():
        data = f()
        print('Dataset:', name)
        print('Nr Topics:', len(data['topic'].unique()))
        nr_instances = len(data)
        print('Instances:', nr_instances)
        for l in data['label'].unique():
            print('Label: {} ({:.2f})'.format(l, len(data[data['label']==l]) / nr_instances ))
