import time
import boto3
client = boto3.client('sagemaker')

CLUSTER_DATASETS = {
    "appen_human":('appen_human_asr', 30, 'text', 'label'),
    "appen_asr":('appen_human_asr', 30, 'transcribed', 'label'),
    "agnews":('agnews', 4, 'text', 'label'),
    "searchsnippets":('searchsnippets', 8, 'text', 'label'),
    "stackoverflow":('stackoverflow', 20, 'text', 'label'),
    "biomedical":('biomedical', 20, 'text', 'label'),
    "tweet":('tweet89', 89, 'text', 'label'),
    "googleT":('googlenews_T', 152, 'text', 'label'),
    "googleS":('googlenews_S', 152, 'text', 'label'),
    "googleTS":('googlenews_TS', 152, 'text', 'label'),
    "googlenews_TS_ctxt_substbertroberta_char_02":('googlenews_TS_ctxt_substbertroberta_char_02', 152, 'text', 'label'),
    "searchsnippets_ctxt_substbertroberta_char_02":('searchsnippets_ctxt_substbertroberta_char_02', 8, 'text', 'label'),
    "stackoverflow_ctxt_substbertroberta_01":('stackoverflow_ctxt_substbertroberta_01', 20, 'text', 'label'),
}


CLUSTER_Augmented_DATASETS_CTXT_20 = {
    "agnews":('agnews_trans_subst_20', 4, 'text', 'label'),
    "searchsnippets":('searchsnippets_trans_subst_20', 8, 'text', 'label'),
    "stackoverflow":('stackoverflow_trans_subst_20', 20, 'text', 'label'),
    "biomedical":('biomedical_trans_subst_20', 20, 'text', 'label'),
    "tweet":('tweet89_trans_subst_20', 89, 'text', 'label'),
    "googleT":('googlenews_T_trans_subst_20', 152, 'text', 'label'),
    "googleS":('googlenews_S_trans_subst_20', 152, 'text', 'label'),
    "googleTS":('googlenews_TS_trans_subst_20', 152, 'text', 'label'),
}

CLUSTER_Augmented_DATASETS_CTXT_CHAR_20 = {
    "agnews":('agnews_trans_subst_20_charswap_20', 4, 'text', 'label'),
    "searchsnippets":('searchsnippets_trans_subst_20_charswap_20', 8, 'text', 'label'),
    "stackoverflow":('stackoverflow_trans_subst_20_charswap_20', 20, 'text', 'label'),
    "biomedical":('biomedical_trans_subst_20_charswap_20', 20, 'text', 'label'),
    "tweet":('tweet89_trans_subst_20_charswap_20', 89, 'text', 'label'),
    "googleT":('googlenews_T_trans_subst_20_charswap_20', 152, 'text', 'label'),
    "googleS":('googlenews_S_trans_subst_20_charswap_20', 152, 'text', 'label'),
    "googleTS":('googlenews_TS_trans_subst_20_charswap_20', 152, 'text', 'label'),
}


CLUSTER_Augmented_DATASETS_CTXT_CHAR_10 = {
    "agnews":('agnews_trans_subst_10_charswap_20', 4, 'text', 'label'),
    "searchsnippets":('searchsnippets_trans_subst_10_charswap_20', 8, 'text', 'label'),
    "stackoverflow":('stackoverflow_trans_subst_10_charswap_20', 20, 'text', 'label'),
    "biomedical":('biomedical_trans_subst_10_charswap_20', 20, 'text', 'label'),
    "tweet":('tweet89_trans_subst_10_charswap_20', 89, 'text', 'label'),
    "googleT":('googlenews_T_trans_subst_10_charswap_20', 152, 'text', 'label'),
    "googleS":('googlenews_S_trans_subst_10_charswap_20', 152, 'text', 'label'),
    "googleTS":('googlenews_TS_trans_subst_10_charswap_20', 152, 'text', 'label'),
}


CLUSTER_Augmented_DATASETS_CTXT_10 = {
    "agnews":('agnews_trans_subst_10', 4, 'text', 'label'),
    "searchsnippets":('searchsnippets_trans_subst_10', 8, 'text', 'label'),
    "stackoverflow":('stackoverflow_trans_subst_10', 20, 'text', 'label'),
    "biomedical":('biomedical_trans_subst_10', 20, 'text', 'label'),
    "tweet":('tweet89_trans_subst_10', 89, 'text', 'label'),
    "googleT":('googlenews_T_trans_subst_10', 152, 'text', 'label'),
    "googleS":('googlenews_S_trans_subst_10', 152, 'text', 'label'),
    "googleTS":('googlenews_TS_trans_subst_10', 152, 'text', 'label'),
}


CLUSTER_Augmented_DATASETS_WDEL_20 = {
    "agnews":('agnews_word_deletion_20', 4, 'text', 'label'),
    "searchsnippets":('searchsnippets_word_deletion_20', 8, 'text', 'label'),
    "stackoverflow":('stackoverflow_word_deletion_20', 20, 'text', 'label'),
    "biomedical":('biomedical_word_deletion_20', 20, 'text', 'label'),
    "tweet":('tweet89_word_deletion_20', 89, 'text', 'label'),
    "googleT":('googlenews_T_word_deletion_20', 152, 'text', 'label'),
    "googleS":('googlenews_S_word_deletion_20', 152, 'text', 'label'),
    "googleTS":('googlenews_TS_word_deletion_20', 152, 'text', 'label'),
}


CLUSTER_Augmented_DATASETS_WDEL_10 = {
    "agnews":('agnews_word_deletion_10', 4, 'text', 'label'),
    "searchsnippets":('searchsnippets_word_deletion_10', 8, 'text', 'label'),
    "stackoverflow":('stackoverflow_word_deletion_10', 20, 'text', 'label'),
    "biomedical":('biomedical_word_deletion_10', 20, 'text', 'label'),
    "tweet":('tweet89_word_deletion_10', 89, 'text', 'label'),
    "googleT":('googlenews_T_word_deletion_10', 152, 'text', 'label'),
    "googleS":('googlenews_S_word_deletion_10', 152, 'text', 'label'),
    "googleTS":('googlenews_TS_word_deletion_10', 152, 'text', 'label'),
}



def wait_till_all_done(base_job_name):
    while True:
        response = client.list_training_jobs(MaxResults=100)
        all_status = [
            item['TrainingJobStatus'] for item in response['TrainingJobSummaries'] if item["TrainingJobName"].startswith(base_job_name)
        ]
        go = all([item != 'InProgress' for item in all_status])
        if go:
            break
        else:
            time.sleep(600) # wait 10 mins

            
