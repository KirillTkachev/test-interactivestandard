import pandas as pd
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score



def evaluate_model(predicts: pd.DataFrame, target: pd.DataFrame) -> dict[str, float]:
    merged_dataframe = predicts.merge(target, how="left", on="file_name")
    return {
        "homogenity_score": homogeneity_score(merged_dataframe["cluster_id"].to_numpy(),
                                              merged_dataframe["predicted"].to_numpy()),
        "completeness_score": completeness_score(merged_dataframe["cluster_id"].to_numpy(),
                                                 merged_dataframe["predicted"].to_numpy()),
        "v_measure_score": v_measure_score(merged_dataframe["cluster_id"].to_numpy(),
                                           merged_dataframe["predicted"].to_numpy())
    }


def create_target_dataframe(labels, images_to_pathes) -> pd.DataFrame:
    predicted_dataframe = pd.DataFrame({'predicted': labels, 'file_name': images_to_pathes.keys()})
    predicted_dataframe['file_name'] = predicted_dataframe['file_name'].astype(str) + '.jpg'
    predicted_dataframe['predicted'] = predicted_dataframe['predicted'].astype(int)
    return predicted_dataframe
