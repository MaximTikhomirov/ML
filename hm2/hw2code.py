import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    sort_idx = np.argsort(feature_vector)
    features_sorted = feature_vector[sort_idx]
    targets_sorted = target_vector[sort_idx]
    
    thresholds = (features_sorted[1:] + features_sorted[:-1]) / 2.0
    
    unique_mask = np.where(features_sorted[1:] != features_sorted[:-1])[0]
    
    thresholds = thresholds[unique_mask]
    left_counts = unique_mask + 1
    
    n_total = len(target_vector)
    
    cumsum_ones = np.cumsum(targets_sorted)
    
    left_ones = cumsum_ones[unique_mask]
    
    n_left = left_counts
    n_right = n_total - n_left
    
    right_ones = np.sum(targets_sorted) - left_ones
    
    p1_left = left_ones / n_left
    p0_left = (n_left - left_ones) / n_left
    
    p1_right = right_ones / n_right
    p0_right = (n_right - right_ones) / n_right
    
    H_left = 1 - p1_left**2 - p0_left**2
    H_right = 1 - p1_right**2 - p0_right**2
    
    ginis = - (n_left / n_total) * H_left - (n_right / n_total) * H_right
    
    if len(ginis) > 0:
        best_idx = np.argmin(ginis)
        threshold_best = thresholds[best_idx]
        gini_best = ginis[best_idx]
    else:
        threshold_best = None
        gini_best = -np.inf
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            if key == 'feature_types':
                setattr(self, '_feature_types', value)
                setattr(self, 'feature_types', value)
            elif key == 'max_depth':
                setattr(self, '_max_depth', value)
                setattr(self, 'max_depth', value)
            elif key == 'min_samples_split':
                setattr(self, '_min_samples_split', value)
                setattr(self, 'min_samples_split', value)
            elif key == 'min_samples_leaf':
                setattr(self, '_min_samples_leaf', value)
                setattr(self, 'min_samples_leaf', value)
        return self
    
    def clone(self):
        return DecisionTree(
            feature_types=self._feature_types.copy() if isinstance(self._feature_types, list) else self._feature_types,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf
        )

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        categories_map_best = None
        
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count if current_count > 0 else 0

                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))
                )
                categories_map = dict(zip(sorted_categories, range(len(sorted_categories))))
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]], dtype=float)
            else:
                raise ValueError

            if np.unique(feature_vector).size < 2:
                continue

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)
            
            if threshold is None:
                continue
                
            current_split = feature_vector < threshold
            left_samples = np.sum(current_split)
            right_samples = len(sub_y) - left_samples
            
            if self._min_samples_leaf is not None:
                if left_samples < self._min_samples_leaf or right_samples < self._min_samples_leaf:
                    continue
            
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = current_split

                if feature_type == "real":
                    threshold_best = threshold
                    categories_map_best = None
                elif feature_type == "categorical":
                    threshold_best = threshold
                    categories_map_best = categories_map.copy()
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            if categories_map_best is not None:
                node["categories_split"] = [
                    k for k, v in categories_map_best.items() if v < threshold_best
                ]
            else:
                node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        
        left_X = sub_X[split]
        left_y = sub_y[split]
        right_X = sub_X[~split]
        right_y = sub_y[~split]
        
        if len(left_y) > 0:
            self._fit_node(left_X, left_y, node["left_child"], depth + 1)
        else:
            node["left_child"]["type"] = "terminal"
            node["left_child"]["class"] = Counter(sub_y).most_common(1)[0][0]
            
        if len(right_y) > 0:
            self._fit_node(right_X, right_y, node["right_child"], depth + 1)
        else:
            node["right_child"]["type"] = "terminal"
            node["right_child"]["class"] = Counter(sub_y).most_common(1)[0][0]

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        f = node["feature_split"]
        if "threshold" in node:
            if x[f] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            cats_left = set(node["categories_split"])
            if x[f] in cats_left:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def score(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return correct / len(y) if len(y) > 0 else 0.0