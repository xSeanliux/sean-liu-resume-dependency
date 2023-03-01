from annotation_object import AnnotationObject
import pickle as pkl

x = AnnotationObject("/Users/liusean/Desktop/Projects/Coding/ML/ForwardLab/SP23/data/pdf/cv_7.pdf")
y = None

with open("./test.pkl", "wb") as handle:
    pkl.dump(x, handle)

with open("./test.pkl", "rb") as handle:
    y = pkl.load(handle)