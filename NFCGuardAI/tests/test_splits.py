import numpy as np
import pandas as pd
import pytest
from nfcguard.dataset.splits import heldout, cv_splits, validate_splits, group_splits, leave_one_tag_out


def test_holdout_partitioning():
    y=np.repeat(np.arange(3),120)
    split=heldout(y,seed=42)
    assert validate_splits(split,len(y))
    assert sum(len(v) for v in split.values())==len(y)
    assert abs(len(split['train'])/len(y)-.7)<.02
    assert abs(len(split['test'])/len(y)-.15)<.02


def test_cv_partitioning():
    y=np.repeat(np.arange(3),120)
    splits=list(cv_splits(y,folds=10,seed=42))
    assert len(splits)==10
    tested=[]
    for fold,split in splits:
        assert validate_splits(split,len(y))
        assert len(split['train'])==252
        tested+=split['test']
    assert sorted(tested)==list(range(len(y)))


def test_loto_is_tag_disjoint():
    tags=np.array(['1']*50+['2']*50+['3']*50)
    sub=['legitimate']*25+['wired']*25
    meta=pd.DataFrame({'tag_id':tags,'subtype':sub*3})
    result=list(leave_one_tag_out(meta))
    assert len(result)==3
    for tag,split in result:
        assert len(split['test'])==50
        assert all(tags[i]==tag for i in split['test'])
        assert all(tags[i]!=tag for i in split['train'])


def test_unknown_groups_refused():
    with pytest.raises(ValueError):
        group_splits(np.array([0,1,0]),['','a','b'])
