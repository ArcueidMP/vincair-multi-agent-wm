import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/e4_navigation'))
from safety_generalization import scenarios,initial_state,audit
class GeneralizationTests(unittest.TestCase):
    def test_scenarios_all_initially_safe(self):
        for g in scenarios():
            for d in [.34,.45]:
                for c in ['angle_orientation','velocity_asymmetry','arrival_offset']:
                    s=initial_state(g,c,d);self.assertGreater(np.linalg.norm(s[0,:2]-s[1,:2]),.30)
    def test_velocity_intervention_preserves_positions(self):
        g=scenarios()[0];a=initial_state(g,'angle_orientation',.34);b=initial_state(g,'velocity_asymmetry',.34)
        np.testing.assert_array_equal(a[:,:2],b[:,:2]);self.assertFalse(np.array_equal(a[:,2:],b[:,2:]))
    def test_no_safe_pool_is_not_false_accept(self):
        raw=np.array([[-1.,0.,1.],[-1.,1.,1.]])
        r=audit(dict(name='x'),raw,np.array([0.,1.]),np.array([.1,.2]),0,1,.25)
        self.assertEqual(r['category'],'no_safe_candidate')
    def test_false_accept_with_safe_option(self):
        raw=np.array([[-5.,0.,1.],[-5.,1.,1.]])
        r=audit(dict(name='x'),raw,np.array([0.,1.]),np.array([.1,.4]),0,1,.25)
        self.assertEqual(r['category'],'risk_false_accept')
if __name__=='__main__':unittest.main()
