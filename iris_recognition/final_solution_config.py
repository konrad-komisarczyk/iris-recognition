# Constants defined for final solution
from iris_recognition.matchers.cosine_similarity_matcher import CosineSimilarityMatcher

FINAL_SOLUTION_MODEL_NAME = "AlexNetFromZero"
FINAL_SOLUTION_MODEL_TAG = "mmu_all_best"
FINAL_SOLUTION_MODEL_NODE = "features.11"
FINAL_SOLUTION_MODEL_EPOCH = 505
FINAL_SOLUTION_MATCHER_CLASS = CosineSimilarityMatcher
FINAL_SOLUTION_MATCHER_THRESHOLD = 0.875
