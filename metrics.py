from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score

def cal_metric(track_label, track_output):
	# print("precision: ", precision_score(track_label.cpu(), track_output.cpu(), average='weighted'))
	# print("recall: ", recall_score(track_label.cpu(), track_output.cpu(), average='weighted'))
	# print("f1-score: ", f1_score(track_label.cpu(), track_output.cpu(), average='weighted'))

	precision = precision_score(track_label.cpu(), track_output.cpu(), average='weighted')
	recall = recall_score(track_label.cpu(), track_output.cpu(), average='weighted')
	f1_scores = f1_score(track_label.cpu(), track_output.cpu(), average='weighted')

	return precision, recall, f1_scores
def plot_conf_matrix(track_label, track_output):
	cm = confusion_matrix(track_label.cpu(), track_output.cpu())
	cm_display = ConfusionMatrixDisplay(cm).plot()
	return 0
