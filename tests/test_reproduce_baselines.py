import importlib.util,unittest
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[1]/'experiments/e4_navigation/reproduce_baselines.py'
spec=importlib.util.spec_from_file_location('reproduction',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1)
class ReproductionTests(unittest.TestCase):
    def test_original_centralized_five_step_regression(self):
        ns=m.context('S1_symmetric','cpu',5)
        m.functions(ns,'03_Centralized_CEM.ipynb',[4],['run_centralized_experiment'])
        original,traj,sep=ns['run_centralized_experiment'](True,42)
        result,data=m.episode('S1_symmetric','centralized_safety',42,'cpu',5)
        np.testing.assert_array_equal(np.stack(traj,axis=1),data['state'][:,:,:2])
        self.assertEqual(original['minimum_separation'],result['minimum_separation'])
        self.assertEqual(result['native_replay_max_abs_error'],0)
    def test_scene_goals_do_not_leak(self):
        for scene in m.SCENARIOS:
            ns=m.context(scene,'cpu',5);env=ns['create_scenario'](2)
            np.testing.assert_array_equal(np.stack([a.goal.state.pos[0].numpy() for a in env.agents]),ns['GOALS'])
    def test_substep_safety_changes_only_diagnostic(self):
        states=np.array([[[0,0,0,0],[1,0,0,0]],[[0,0,0,0],[1,0,0,0]]],np.float32)
        subs=np.array([[[.45,0,0,0],[.55,0,0,0]],states[-1]],np.float32)
        result=m.calculate_metrics(states,subs,states[-1,:,:2],[.05,.05])
        self.assertTrue(result['joint_success']);self.assertFalse(result['loss_of_separation'])
        self.assertTrue(result['substep_loss_of_separation']);self.assertFalse(result['safe_joint_success'])
    def test_priority_matches_notebook_positions(self):
        ns=m.context('S1_symmetric','cpu',8);ns.update(KP=1.5,KD=1.)
        m.functions(ns,'01_PD_Baselines.ipynb',[11],['run_experiment'])
        original,traj,_=ns['run_experiment']('priority')
        result,data=m.episode('S1_symmetric','priority_pd',42,'cpu',8)
        np.testing.assert_array_equal(np.stack(traj,axis=1),data['state'][:,:,:2])
if __name__=='__main__':unittest.main()
