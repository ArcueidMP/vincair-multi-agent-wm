import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/e4_navigation'))
from conflict_ranking import preference_order,pairs,layouts
class ConflictTests(unittest.TestCase):
    def test_safe_preferred_and_threshold(self):
        order=preference_order(np.array([0.,10.,3.]),np.array([.299,.30,.35]),np.zeros(3))
        np.testing.assert_array_equal(order,[2,1,0])
    def test_all_unsafe_uses_robust_fallback(self):
        order=preference_order(np.array([0.,100.,1.]),np.array([.1,.29,.2]),np.ones(3))
        self.assertEqual(order[0],1)
    def test_ties_not_training_preferences(self):
        p,k=pairs(np.ones(3),np.ones(3),np.zeros(3));self.assertEqual(p.shape,(0,2))
    def test_pairs_direction_agrees_with_safety_order(self):
        c=np.array([0.,10.,3.,1.]);s=np.array([.1,.30,.35,.2]);e=np.array([1.,0.,0.,.1]);order=preference_order(c,s,e)
        rank=np.argsort(order);p,k=pairs(c,s,e);self.assertTrue((rank[p[:,0]]<rank[p[:,1]]).all());self.assertTrue((k==0).any())
    def test_geometry_ids_cannot_cross_splits(self):
        rows=layouts();self.assertEqual(len({q['geometry_id'] for q in rows}),18)
        self.assertEqual({s:sum(q['split']==s for q in rows) for s in ['train','validation','test']},{'train':12,'validation':3,'test':3})
if __name__=='__main__':unittest.main()
