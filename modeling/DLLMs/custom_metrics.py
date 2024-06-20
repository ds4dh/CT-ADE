import datasets
from sklearn.metrics import balanced_accuracy_score
import evaluate

_DESCRIPTION = """
Balanced accuracy computes the average of recall obtained on each class and is especially useful for imbalanced datasets.
It is defined as the average of recall obtained on each class.
The best value is 1 and the worst value is 0, particularly when 'adjusted=False'.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    sample_weight (`list` of `float`): Sample weights, defaults to None.
    adjusted (`boolean`): Adjusts the score to account for chance, defaults to False.

Returns:
    balanced_accuracy (`float`): Balanced accuracy score. Minimum possible value is 0. Maximum possible value is 1.0.
"""

_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BalancedAccuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html"
            ],
        )

    def _compute(self, predictions, references, sample_weight=None, adjusted=False):
        return {
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    references,
                    predictions,
                    sample_weight=sample_weight,
                    adjusted=adjusted,
                )
            )
        }