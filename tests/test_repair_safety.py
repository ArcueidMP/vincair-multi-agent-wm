import sys, unittest, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/e4_navigation'))
import repair_safety as r

class RepairTests(unittest.TestCase):
    def setUp(self):
        self.s=np.array([[-.2,.1,.1,.2],[.3,-.1,-.2,.1]],np.float32)
        self.g=np.array([[1.,.5],[-1.,-.5]],np.float32)
        self.a=np.random.default_rng(4).uniform(-.4,.4,(64,8,2,2)).astype(np.float32)
    def test_rigid_coordinates(self):
        angle=.73;q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],np.float32)
        s=self.s.copy();s[:,:2]=s[:,:2]@q.T+np.array([.3,-.7]);s[:,2:]=s[:,2:]@q.T
        np.testing.assert_allclose(r.invariant_features(s,self.g@q.T+[.3,-.7],self.a@q.T),r.invariant_features(self.s,self.g,self.a),atol=5e-7)
    def test_agent_order_views(self):
        x=r.invariant_features(self.s,self.g,self.a)
        z=r.invariant_features(self.s[::-1],self.g[::-1],self.a[:,:,::-1])
        np.testing.assert_array_equal(x[:,::-1],z)
    def test_proposals_keep_original(self):
        a=r.proposals(self.s,self.a)
        np.testing.assert_array_equal(a[:64],self.a);self.assertEqual(a.shape,(68,8,2,2))
        self.assertEqual(abs(a).max(),1);np.testing.assert_array_equal(a[64],0)
    def test_risk_sep_disagreement(self):
        raw=np.array([[[-5,0,.5],[-4,1,1.2]]],np.float32)
        idx,fb=r.choose(raw,0,1,.25,0);self.assertEqual(idx[0],1);self.assertFalse(fb[0])
    def test_fallback_minimum_risk(self):
        raw=np.array([[[3,0,2],[2,1,1],[2,0,1.5]]],np.float32)
        idx,fb=r.choose(raw,0,1,.1,0);self.assertEqual(idx[0],2);self.assertTrue(fb[0])
    def test_confirmation_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):r.load(Path(tmp),'confirmation','raw')
    def test_split_whole_geometry(self):
        rows=r.geometries();self.assertEqual(len(rows),44);self.assertEqual(len({q['base_id'] for q in rows}),44)
        self.assertEqual([sum(q['split']==s for q in rows) for s in ['train','validation','confirmation']],[24,8,12])

if __name__=='__main__':unittest.main()
