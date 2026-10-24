import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/e4_navigation'))
import robust_navigation as r
import shared_navigation_model as shared

class RobustnessTests(unittest.TestCase):
    def test_complete_paired_layout(self):
        rows=r.layouts();self.assertEqual(len(rows),24);self.assertEqual(len({x['base_id'] for x in rows}),6)
        for gid in {x['base_id'] for x in rows}:
            group=[x for x in rows if x['base_id']==gid];self.assertEqual({x['condition'] for x in group},set(r.CONDITIONS))
            for x in group[1:]:np.testing.assert_array_equal(x['goals'],group[0]['goals'])
    def test_initial_safety_and_nonterminal(self):
        for geo in r.layouts():
            self.assertGreater(np.linalg.norm(np.diff(geo['starts'],axis=0)),.3)
            self.assertTrue((np.linalg.norm(np.array(geo['goals'])-geo['starts'],axis=-1)>.12).all())
    def test_terminal_condition_is_goal_approach(self):
        for geo in r.layouts():
            if geo['condition']=='terminal_neighborhood':np.testing.assert_allclose(np.linalg.norm(np.array(geo['goals'])-geo['starts'],axis=-1),.2,atol=1e-6)
    def test_portable_loader_contract(self):
        models,obj=shared.load();self.assertEqual(obj['kind'],'direct current-state/action scorer; NOT Navigation LeWM')
        self.assertEqual(models['invariant_repair']['checkpoint']['policy'],dict(threshold=.5,margin=0.))
        self.assertFalse(models['invariant_repair']['net'].training)

if __name__=='__main__':unittest.main()
