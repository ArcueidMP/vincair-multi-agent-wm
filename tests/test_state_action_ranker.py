import sys,tempfile,unittest,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/e4_navigation'))
from state_action_ranker import features,select,load,Scorer
class DirectScorerTests(unittest.TestCase):
    def test_feature_contract(self):
        x=features(np.zeros((2,4)),np.ones((2,2)),np.zeros((64,8,2,2)))
        self.assertEqual(x.shape,(64,44));self.assertTrue(np.array_equal(x[:,8:12],np.ones((64,4))))
    def test_future_labels_rejected_as_feature_argument(self):
        with self.assertRaises(TypeError):features(np.zeros((2,4)),np.ones((2,2)),np.zeros((64,8,2,2)),true_cost=np.ones(64))
    def test_test_guard_precedes_file_io(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):load(Path(d),'test',[])
    def test_all_predicted_unsafe_fallback(self):
        raw=np.array([[[10.,0.,.1],[10.,3.,.5]]]);idx,fallback,*_=select(raw,0,1,.1)
        self.assertEqual(idx[0],1);self.assertTrue(fallback[0])
    def test_test_guard_checks_checkpoint_hash_before_data_io(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'state_action_scorer.pt').write_bytes(b'not the frozen checkpoint')
            (p/'freeze.json').write_text(json.dumps(dict(status='frozen',checkpoint_sha256='wrong',protocol_sha256='wrong')))
            with self.assertRaises(AssertionError):load(p/'nonexistent_data','test',[],p/'freeze.json')
    def test_predicted_safe_cost_choice(self):
        raw=np.array([[[-10.,2.,.4],[-10.,1.,.4],[10.,-5.,.1]]]);idx,*_=select(raw,0,1,.25);self.assertEqual(idx[0],1)
    def test_parameter_count(self):self.assertEqual(sum(p.numel() for p in Scorer().parameters()),5059)
if __name__=='__main__':unittest.main()
