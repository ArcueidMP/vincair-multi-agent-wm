import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/e4_navigation'))
import closed_loop_pilot as p
import conflict_ranking as cr

class ClosedLoopTests(unittest.TestCase):
    def test_pd_matches_native_baseline_formula(self):
        s=np.array([[0,0,.2,-.1],[0,0,-.1,.2]],np.float32);g=np.array([[.5,.5],[-.5,-.5]],np.float32)
        np.testing.assert_allclose(p.pd(s,g),[[.55,.85],[-.65,-.95]],atol=1e-7)
    def test_pd_clips_component_limits(self):
        np.testing.assert_array_equal(p.pd(np.zeros((2,4)),np.array([[10,-10],[-10,10]])),[[1,-1],[-1,1]])
    def test_paired_candidate_generation(self):
        geo=p.geometries()[0];s=np.concatenate([geo['starts'],geo['velocity']],1).astype(np.float32);g=np.array(geo['goals'],np.float32)
        ns=cr.namespace('S1_symmetric',g);a=p.candidates(ns,s,g,geo['directions'],3);b=p.candidates(ns,s,g,geo['directions'],3)
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(72,8,2,2));self.assertLessEqual(abs(a).max(),1)
        np.testing.assert_array_equal(a[68,0],p.pd(s,g))
    def test_exact_goal_candidate_direction_finite(self):
        geo=p.geometries()[0];g=np.array(geo['goals'],np.float32);s=np.concatenate([g,np.zeros((2,2))],1)
        a=p.candidates(cr.namespace('S1_symmetric',g),s,g,geo['directions'],7)
        self.assertTrue(np.isfinite(a).all())
    def test_fresh_geometries_initially_safe(self):
        geos=p.geometries();self.assertEqual(len(geos),6);self.assertEqual(len({g['name'] for g in geos}),6)
        for g in geos:self.assertGreater(np.linalg.norm(np.diff(g['starts'],axis=0)),.3)

if __name__=='__main__':unittest.main()
