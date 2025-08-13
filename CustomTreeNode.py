class TreeNode:
    """
    Custom TreeNode class for Decision Tree implementation
    """
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain):
        """
        Initialize a tree node
        
        Args:
            data: The data associated with this node
            feature_idx: Index of the feature used for splitting
            feature_val: Value of the feature used for splitting
            prediction_probs: Prediction probabilities for this node
            information_gain: Information gain from the split
        """
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        
        # Tree structure
        self.left = None
        self.right = None
        
        # Calculate feature importance (can be used for feature importance analysis)
        self.feature_importance = information_gain
    
    def node_def(self):
        """Return a string representation of the node definition"""
        if self.left is None and self.right is None:
            return f"Leaf: {self.prediction_probs}"
        else:
            return f"Feature {self.feature_idx} < {self.feature_val:.3f}" 