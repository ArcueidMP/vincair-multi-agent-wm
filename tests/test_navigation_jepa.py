import unittest
import torch
from vmas_navigation import data
from vmas_navigation.model import NavigationJEPA
from vmas_navigation.pilot import normalization,seed_training

class NavigationPilotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(0)
    def test_geometry_group_split(self):
        rows=data.layouts();self.assertEqual(len(rows),16);self.assertEqual(sum(r['split']=='train' for r in rows),12)
        self.assertEqual(len({r['geometry_id'] for r in rows}),16)
    def test_exact_temporal_window_alignment(self):
        s=torch.arange(21.).reshape(1,21,1,1).expand(2,21,2,4).clone();a=torch.arange(20.).reshape(1,20,1,1).expand(2,20,2,2).clone()
        x,y,e,t=data.sample(s,a,torch.tensor([0]),8,3,3,torch.Generator().manual_seed(3))
        self.assertEqual(x.shape,(8,6,2,4));self.assertTrue((e==0).all())
        self.assertTrue(torch.equal(x[:,1:,0,0]-x[:,:-1,0,0],torch.ones(8,5)))
        self.assertTrue(torch.equal(y[:,:,0,0],x[:,:-1,0,0]))
    def test_normalization_uses_only_training(self):
        s=torch.randn(4,10,2,4);s[2:]+=10000
        mean,std,delta=normalization(s,torch.tensor([0,1]));torch.testing.assert_close(mean,s[:2].reshape(-1,4).mean(0))
        self.assertTrue((std>=.01).all());self.assertTrue((delta>=.02).all())
    def test_agent_permutation_equivariance(self):
        m=NavigationJEPA().eval();s=torch.randn(2,3,2,4);a=torch.randn(2,3,2,2)
        with torch.no_grad():
            z=m.encode(s);pred=m.predict(z,a);other=m.predict(m.encode(s.flip(2)),a.flip(2))
        torch.testing.assert_close(pred.flip(1),other,atol=1e-6,rtol=1e-5)
    def test_dynamic_three_agent_shape_is_only_api_test(self):
        m=NavigationJEPA().eval();s=torch.randn(2,3,3,4);a=torch.randn(2,3,3,2)
        with torch.no_grad():pred=m.predict(m.encode(s),a)
        self.assertEqual(m.state_head(pred).shape,(2,3,4))
    def test_finite_loss_and_gradient(self):
        m=NavigationJEPA();loss,_=m.loss(torch.randn(8,6,2,4),torch.randn(8,5,2,2),torch.ones(4))
        self.assertTrue(torch.isfinite(loss));loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    def test_training_seed_restored_after_native_environment(self):
        ns=data.native.context('S1_symmetric','cpu',5);ns['create_scenario'](4)
        self.assertEqual(seed_training(0),dict(cpu_initial_seed=0,cuda_initial_seed=0))
    def test_native_collection_observation_and_replay(self):
        d,rows=data.collect(data.layouts()[:1],4)
        self.assertEqual(d['state'].shape,(4,5,2,4));self.assertEqual(len(rows),4)
        self.assertTrue(torch.equal(d['state'],d['native_observation'][...,:4]));self.assertTrue(torch.equal(d['substep_state'][:,1::2],d['state'][:,1:]))

if __name__=='__main__':unittest.main()
