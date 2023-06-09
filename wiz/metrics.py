from sklearn import metrics


def metrics_info_print(head, Y_target, predict, tn, fp):
    # print(f"\n{head}_target:\n{list(Y_target)}")
    # print(f"{head}_predict:\n{list(predict)}")
    print(f"\nacc over {head} set: {metrics.accuracy_score(Y_target, predict)}")
    print(f"precision over {head} set: {metrics.precision_score(Y_target, predict)}")
    print(f"recall over {head} set: {metrics.recall_score(Y_target, predict)}")
    print(f"F1 over {head} set: {metrics.f1_score(Y_target, predict)}")
    print(f"specificity over {head} set: {tn / (tn+fp)}")


def metrics_info(head, Y_target, predict, tn, fp):
    return f"""\n{head}_target:\n{list(Y_target)}
{head}_predict:\n{list(predict)}
acc over {head} set: {metrics.accuracy_score(Y_target, predict)}
precision over {head} set: {metrics.precision_score(Y_target, predict)}
recall over {head} set: {metrics.recall_score(Y_target, predict)}
F1 over {head} set: {metrics.f1_score(Y_target, predict)}
specificity over {head} set: {tn / (tn+fp)}"""
