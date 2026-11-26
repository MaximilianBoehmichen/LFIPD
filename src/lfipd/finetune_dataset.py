from lfipd.eval_dataset import EvalDataset


class FinetuneDataset(EvalDataset):
    """The Dataset of samples with missing modalities used for finetuning.

    Currently, the intended behavior is already supported by :class:`EvalDataset`.
    """
