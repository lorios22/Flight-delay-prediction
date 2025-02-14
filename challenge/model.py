import pandas as pd
from typing import Tuple, Union, List
import xgboost as xgb


class DelayModel:
    """A model for predicting flight delays.

    This model uses XGBoost to predict whether a flight will be delayed by 
    more than 15 minutes. It preprocesses raw flight data into features and 
    handles class imbalance during training.
    """

    def __init__(self) -> None:
        """Initialize the model with predefined important features."""
        self._model = None
        self.top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """Prepare raw data for training or prediction.

        Args:
            data: Raw flight data containing features like OPERA, TIPOVUELO, MES
            target_column: Name of target column. If provided, calculates labels

        Returns:
            Either a tuple of (features, target) if target_column is provided,
            or just features DataFrame if target_column is None
        """
        if target_column and target_column not in data.columns:
            data[target_column] = self._calculate_delay_labels(data)

        features = self._create_feature_matrix(data)

        if target_column:
            target = pd.DataFrame(data[target_column], columns=[target_column])
            return features, target

        return features

    def _calculate_delay_labels(self, data: pd.DataFrame) -> pd.Series:
        """Calculate binary delay labels from flight timestamps.

        Args:
            data: DataFrame containing Fecha-I and Fecha-O columns

        Returns:
            Series of binary delay labels (1 if delayed >15 min, 0 otherwise)
        """
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'])

        min_diff = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
        threshold_in_minutes = 15

        return (min_diff > threshold_in_minutes).astype(int)

    def _create_feature_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix from categorical variables.

        Args:
            data: DataFrame containing OPERA, TIPOVUELO and MES columns

        Returns:
            DataFrame with one-hot encoded features, filtered to top 10 important
        """
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        return features.reindex(columns=self.top_10_features, fill_value=0)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train the model on preprocessed data.

        Args:
            features: Preprocessed feature matrix
            target: DataFrame containing 'delay' column with binary labels
        """
        scale = self._calculate_class_weight(target)

        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale,
            max_depth=3,
            min_child_weight=5
        )
        self._model.fit(features, target['delay'])

    def _calculate_class_weight(self, target: pd.DataFrame) -> float:
        """Calculate class weight to handle imbalanced classes.

        Args:
            target: DataFrame containing 'delay' column

        Returns:
            Weight to scale positive class during training
        """
        n_y0 = len(target[target['delay'] == 0])
        n_y1 = len(target[target['delay'] == 1])
        return n_y0 / n_y1

    def predict(self, features: pd.DataFrame) -> List[int]:
        """Generate predictions for preprocessed features.

        Args:
            features: Preprocessed feature matrix

        Returns:
            List of binary predictions (1 for delay, 0 for no delay)
        """
        if self._model is None:
            return [0] * features.shape[0]

        predictions = self._model.predict(features)
        return predictions.tolist()
