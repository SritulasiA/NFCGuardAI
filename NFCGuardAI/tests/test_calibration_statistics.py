import numpy as np
from nfcguard.training.calibration import fit_temperature,temperature_predict,reliability
from nfcguard.evaluation.metrics import binary_metrics,summarize_folds
from nfcguard.evaluation.statistics import corrected_resampled_t,holm_bonferroni,mcnemar_paired


def test_temperature():
    z=np.array([-4,-3,-2,2,3,4],dtype=float)
    y=np.array([0,0,1,0,1,1],dtype=int)
    t=fit_temperature(z,y)
    assert .05<=t<=20
    p=temperature_predict(z,t)
    assert p.shape==(6,)
    assert np.isfinite(p).all()
    cal=reliability(y,p)
    assert 0<=cal['ece']<=1


def test_binary_metrics():
    y=[0,0,1,1]
    p=[.1,.7,.6,.9]
    result=binary_metrics(y,p,subtype=[0,0,1,2])
    assert result['tp']==2
    assert result['fp']==1
    assert result['tn']==1
    assert result['fn']==0
    assert result['subtype_recall']['wired']==1.


def test_statistical_tests():
    a=np.linspace(.91,.95,10)
    b=a-.02+np.linspace(-.006,.006,10)
    stats=corrected_resampled_t(a,b,.9,.1)
    assert stats['n']==10
    assert stats['mean_difference']>0
    adj=holm_bonferroni([.01,.03,.08])
    assert len(adj)==3
    assert adj[0]>=.01
    mc=mcnemar_paired([True,True,True,False],[True,False,False,False])
    assert mc['a_only']==2
