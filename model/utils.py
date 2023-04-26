all_labels = ['discard', 'merge', 'pop', 'subordinate']
n_classes = len(all_labels)
label_idx = {lab: int(i) for i, lab in enumerate(all_labels)}
idx_label = {int(i): lab for i, lab in enumerate(all_labels)}