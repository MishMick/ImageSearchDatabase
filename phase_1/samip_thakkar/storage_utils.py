
from feature_store import FeatureStorage, ModelType

# Pickle the HOG
storage_path = "C:/ASU/Sem 1/MWDB/ASU MWDB PROJECT/Project/Project Files/store/"
feature_storage = FeatureStorage(storage_path)
feature_storage.store_to_pkl(ModelType.HOG)
#feature_storage.store_to_pkl(ModelType.COLOR_MOMENTS)
