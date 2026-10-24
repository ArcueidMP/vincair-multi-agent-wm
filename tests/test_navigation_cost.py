import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/e4_navigation'))
import repair_navigation_cost as m
import conflict_ranking as cr
import repair_safety as repair

class CostTests(unittest.TestCase):
    def setUp(self):
        self.s=np.array([[-1.,0,.1,0],[1.,0,-.1,0]],np.float32);self.g=np.array([[1.,0],[-1.,0]],np.float32)
        self.actions=np.random.default_rng(9).uniform(-.1,.1,(72,8,2,2)).astype(np.float32)
    def test_analytic_matches_native_without_contact(self):
        torch.set_num_threads(1);ns=cr.namespace('S1_symmetric',self.g);path,_=repair.rollout(ns,self.s,torch.tensor(self.actions,device='cuda'))
        expected=5*np.linalg.norm(path[:,-1,:,:2].cpu().numpy()-self.g,axis=-1).sum(-1)+.01*(self.actions**2).mean((1,2,3))
        np.testing.assert_allclose(m.analytic(self.s,self.g,self.actions)[0],expected,atol=2e-6,rtol=1e-6)
    def test_feature_contract(self):
        x=m.correction_features(self.s,self.g,self.actions,np.zeros((72,3),np.float32),0,1)
        self.assertEqual(x.shape,(72,92));self.assertTrue(np.isfinite(x).all())
    def test_swap_consistency(self):
        raw=np.zeros((72,3),np.float32)
        x=m.correction_features(self.s,self.g,self.actions,raw,0,1);y=m.correction_features(self.s[::-1],self.g[::-1],self.actions[:,:,::-1],raw,0,1)
        np.testing.assert_allclose(x,y,atol=2e-6,rtol=1e-6)
    def test_risk_and_separation_heads_remain_frozen(self):
        raw=np.tile(np.array([-3.,4.,1.2],np.float32),(72,1));original=m.ORIGINAL_DECIDE
        try:
            m.ORIGINAL_DECIDE=lambda *args:(0,False,raw.copy())
            model=dict(progress_mode='analytic',checkpoint=dict(cost_mean=0,cost_std=1,policy=dict(threshold=.5,margin=0)))
            _,_,changed=m.decision(model,self.s,self.g,self.actions)
            np.testing.assert_array_equal(changed[:,[0,2]],raw[:,[0,2]])
        finally:m.ORIGINAL_DECIDE=original
    def test_fresh_confirmation_geometry_count(self):
        rows=m.geometries();self.assertEqual(len(rows),6)
        self.assertEqual(len({r['name'] for r in rows}),6)
        for r in rows:self.assertGreater(np.linalg.norm(np.diff(r['starts'],axis=0)),.3)

if __name__=='__main__':unittest.main()
