import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/e4_navigation'))
import navigation_pressure as pressure
from verify_shared_reproduction import compare

class ReadinessTests(unittest.TestCase):
    def row(self):return dict(geometry='g',arm='a',seed=1,joint_success=True,safe_joint_success=True,substep_loss_of_separation=False,substep_collision_proxy=False,steps=5,substep_minimum_separation=.31)
    def test_reproduction_ignores_latency(self):
        a=self.row();b=dict(a,mean_planning_ms=100000);self.assertEqual(compare([a],[b]),[])
    def test_reproduction_detects_changed_outcome(self):
        a=self.row();b=dict(a,safe_joint_success=False);self.assertEqual(len(compare([a],[b])),1)
    def test_reproduction_rejects_missing_episode(self):
        with self.assertRaises(ValueError):compare([self.row()],[])
    def test_reproduction_rejects_duplicate(self):
        with self.assertRaises(ValueError):compare([self.row(),self.row()],[self.row()])
    def test_pressure_layout_contract(self):
        rows=pressure.layouts();self.assertEqual(len(rows),12);self.assertEqual(len({r['base_id'] for r in rows}),6)
        for r in rows:np.testing.assert_allclose(np.linalg.norm(np.diff(r['starts'],axis=0)),r['initial_separation'],atol=1e-6)

if __name__=='__main__':unittest.main()
